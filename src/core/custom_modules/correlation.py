import torch
import torch.nn as nn
import re
import cupy

class Stream:
    ptr = torch.cuda.current_stream().cuda_stream


def cupy_kernel(strFunction, objectVariables):

    strKernel = globals()[strFunction]

    while True:
        objectMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

        if objectMatch is None:
            break

        intArg = int(objectMatch.group(2))

        strTensor = objectMatch.group(4)
        intSizes = objectVariables[strTensor].size()

        strKernel = strKernel.replace(objectMatch.group(), str(intSizes[intArg]))


    while True:
        objectMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objectMatch is None:
            break

        intArgs = int(objectMatch.group(2))
        strArgs = objectMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objectVariables[strTensor].stride()
        strIndex = [ '((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs) ]

        strKernel = strKernel.replace(objectMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')

    return strKernel



@cupy.util.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)


kernel_Correlation_rearrange = '''
    extern "C" __global__ void kernel_Correlation_rearrange(
                        const int n,
                        const int pad_size,
                        const float* input,
                        float* output)
    {
    
    int intIndex = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (intIndex >= n) { 
        return;
        }
        
    int intSample = blockIdx.z;
    int intChannel = blockIdx.y;
    
    float dblValue = input[(intSample * SIZE_1(input) + intChannel) * SIZE_2(input) * SIZE_3(input) + intIndex];
    
    __syncthreads();
    
    int intPaddedX = intIndex % SIZE_3(input) + pad_size;
    int intPaddedY = intIndex / SIZE_3(input) + pad_size;
    int intRearrange = intPaddedY * SIZE_2(output) + intPaddedX;
    int out_index = (intSample * SIZE_1(output) * SIZE_2(output) + intRearrange) * SIZE_3(output) + intChannel;
    
    output[out_index] = dblValue;
    }
'''

kernel_Correlation_updateOutput = '''

    extern "C" __global__ void kernel_Correlation_updateOutput(
                        const int num,
                        const int topwidth,
                        const int topheight,
                        const int topchannels,
                        const int topcount,
                        const int max_displacement,
                        const int neighborhood_grid_radius, 
                        const int neighborhood_grid_width, 
                        const int kernel_radius, 
                        const int kernel_size, 
						const int stride1, 
						const int stride2,
						const int bottomwidth, 
						const int bottomheight, 
						const int bottomchannels,
                        const float* bottom0,
                        const float* bottom1,
                        float* top)
                        
    {
		extern __shared__ char patch_data_char[];
		float* patch_data = (float *)patch_data_char;
	
		// First (upper left) position of kernel upper-left corner in current center position of neighborhood in image 1
		int x1 = blockIdx.x*stride1 + max_displacement;
		int y1 = blockIdx.y*stride1 + max_displacement;
		int item = blockIdx.z;
		int ch_off = threadIdx.x;
	
		// Load 3D patch into shared shared memory
		for(int j = 0; j < kernel_size; j++) 
		{ // HEIGHT
			for(int i = 0; i < kernel_size; i++) 
			{ // WIDTH
				int ji_off = ((j * kernel_size) + i) * bottomchannels;
				for(int ch = ch_off; ch < bottomchannels; ch += 32) 
				{ // CHANNELS
					int idx1 = ((item * bottomheight + y1+j) * bottomwidth + x1+i) * bottomchannels + ch;
					int idxPatchData = ji_off + ch;
					patch_data[idxPatchData] = bottom0[idx1];
				}
			}
		}
	
		__syncthreads();
	
		__shared__ float sum[32];
	
		// Compute correlation
		for(int top_channel = 0; top_channel < topchannels; top_channel++) 
		{
			sum[ch_off] = 0;
	
			int s2o = (top_channel % neighborhood_grid_width - neighborhood_grid_radius) * stride2;
			int s2p = (top_channel / neighborhood_grid_width - neighborhood_grid_radius) * stride2;
	
			for(int j = 0; j < kernel_size; j++)
			{ // HEIGHT
				for(int i = 0; i < kernel_size; i++) 
				{ // WIDTH
					int ji_off = ((j * kernel_size) + i) * bottomchannels;
					for(int ch = ch_off; ch < bottomchannels; ch += 32) 
					{ // CHANNELS
						int x2 = x1 + s2o;
						int y2 = y1 + s2p;
	
						int idxPatchData = ji_off + ch;
						int idx2 = ((item * bottomheight + y2+j) * bottomwidth + x2+i) * bottomchannels + ch;
	
						sum[ch_off] += patch_data[idxPatchData] * bottom1[idx2];
					}
				}
			}
	
			__syncthreads();
	
			if(ch_off == 0) 
			{
				float total_sum = 0;
				for(int idx = 0; idx < 32; idx++)
				{
					total_sum += sum[idx];
				}
				
			const int sumelems = kernel_size*kernel_size*bottomchannels;
			const int index = ((top_channel*topheight + blockIdx.y)*topwidth)+blockIdx.x;
			top[index + item*topcount] = total_sum / (float)sumelems;
			}
		}
		
	}
'''


kernel_Correlation_updateGradFirst = '''

	extern "C" __global__ void kernel_Correlation_updateGradFirst( 
									int num, 
									int item, 
									int topwidth, 
									int topheight, 
									int topchannels,
									int max_displacement, 
									int neighborhood_grid_radius, 
									int neighborhood_grid_width, 
									int kernel_radius, 
									int stride1, 
									int stride2,
									int bottomwidth, 
									int bottomheight, 
									int pbottomwidth, 
									int pbottomheight, 
									int bottomchannels, 
									int bottomcount, 
									int pad_size,
									float *bottom0diff, 
									const float *bottom1, 
									const float *topdiff)
									
	{
	
		#define ROUND_OFF 50000
		
		for (int index = blockIdx.x*blockDim.x + threadIdx.x; index < bottomcount;
								index += blockDim.x * gridDim.x)
		{
			int n = index % bottomchannels; //channels
			int l = (index / bottomchannels) % bottomwidth + pad_size; //w-pos
			int m = (index / bottomchannels / bottomwidth) % bottomheight + pad_size; //h-pos
	
			//Get X,Y ranges and clamp
			// round_off is a trick to enable integer division with ceil, even for negative numbers
			// We use a large offset, for the inner part not to become negative.
			const int round_off = ROUND_OFF;
			const int round_off_s1 = stride1 * round_off;
	
			// We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
			int xmin = (l - 2*kernel_radius - max_displacement + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1
			int ymin = (m - 2*kernel_radius - max_displacement + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1
	
			// Same here:
			int xmax = (l - max_displacement + round_off_s1) / stride1 - round_off; // floor (l - max_displacement) / stride1
			int ymax = (m - max_displacement + round_off_s1) / stride1 - round_off; // floor (m - max_displacement) / stride1
	
	
			float sum = 0;
			if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
			{
				xmin = max(0,xmin);
				xmax = min(topwidth-1,xmax);
	
				ymin = max(0,ymin);
				ymax = min(topheight-1,ymax);
	
				for(int p = -neighborhood_grid_radius; p <= neighborhood_grid_radius; p++) 
				{
					for(int o = -neighborhood_grid_radius; o <= neighborhood_grid_radius; o++) 
					{
	
						// Get bottom1 data:
						int s2o = stride2 * o;
						int s2p = stride2 * p;
						int idxbot1 = ((item * pbottomheight + (m+s2p)) * pbottomwidth + (l+s2o)) * bottomchannels + n;
						float bot1tmp = bottom1[idxbot1]; // bottom1[l+s2o,m+s2p,n]
	
						// Index offset for topdiff in following loops:
						int op = (p+neighborhood_grid_radius) * neighborhood_grid_width + (o+neighborhood_grid_radius); // index [o,p]
						int idxopoffset = (item * topchannels + op);
	
						for(int y = ymin; y <= ymax; y++) 
						{
							for(int x = xmin; x <= xmax; x++) 
							{
								int idxtopdiff = (idxopoffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
								sum += topdiff[idxtopdiff] * bot1tmp;
							}
						}
					}
				}
			}
		
			const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
			const int bot0index = ((n * bottomheight) + (m-pad_size)) * bottomwidth + (l-pad_size);
			bottom0diff[bot0index + item*bottomcount] = sum / (float)sumelems;
		}
	}

'''

kernel_Correlation_updateGradSecond = '''

	extern "C" __global__ void kernel_Correlation_updateGradSecond( 
									int num, 
									int item, 
									int topwidth, 
									int topheight, 
									int topchannels,
									int max_displacement, 
									int neighborhood_grid_radius, 
									int neighborhood_grid_width, 
									int kernel_radius, 
									int stride1, 
									int stride2,
									int bottomwidth, 
									int bottomheight, 
									int pbottomwidth, 
									int pbottomheight, 
									int bottomchannels, 
									int bottomcount, 
									int pad_size,
									float *bottom1diff, 
									const float *bottom0, 
									const float *topdiff)
									
	{
	
		#define ROUND_OFF 50000
			
		for (int index = blockIdx.x*blockDim.x + threadIdx.x; index < bottomcount;
								index += blockDim.x * gridDim.x)
		
		{
									
			int n = index % bottomchannels; //channels
			int l = (index / bottomchannels) % bottomwidth + pad_size; //w-pos
			int m = (index / bottomchannels / bottomwidth) % bottomheight + pad_size; //h-pos
	
			// round_off is a trick to enable integer division with ceil, even for negative numbers
			// We use a large offset, for the inner part not to become negative.
			const int round_off = ROUND_OFF;
			const int round_off_s1 = stride1 * round_off;
	
			float sum = 0;
			for(int p = -neighborhood_grid_radius; p <= neighborhood_grid_radius; p++) 
			{
				for(int o = -neighborhood_grid_radius; o <= neighborhood_grid_radius; o++) 
				{
					int s2o = stride2 * o;
					int s2p = stride2 * p;
	
					//Get X,Y ranges and clamp
					// We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
					int xmin = (l - 2*kernel_radius - max_displacement - s2o + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
					int ymin = (m - 2*kernel_radius - max_displacement - s2p + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
	
					// Same here:
					int xmax = (l - max_displacement - s2o + round_off_s1) / stride1 - round_off; // floor (l - max_displacement - s2o) / stride1
					int ymax = (m - max_displacement - s2p + round_off_s1) / stride1 - round_off; // floor (m - max_displacement - s2p) / stride1
	
					if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
					{
						xmin = max(0,xmin);
						xmax = min(topwidth-1,xmax);
	
						ymin = max(0,ymin);
						ymax = min(topheight-1,ymax);
	
						// Get bottom0 data:
						int idxbot0 = ((item * pbottomheight + (m-s2p)) * pbottomwidth + (l-s2o)) * bottomchannels + n;
						float bot0tmp = bottom0[idxbot0]; // bottom1[l+s2o,m+s2p,n]
	
						// Index offset for topdiff in following loops:
						int op = (p+neighborhood_grid_radius) * neighborhood_grid_width + (o+neighborhood_grid_radius); // index [o,p]
						int idxOpOffset = (item * topchannels + op);
	
						for(int y = ymin; y <= ymax; y++) 
						{
							for(int x = xmin; x <= xmax; x++) 
							{
								int idxtopdiff = (idxOpOffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
								sum += topdiff[idxtopdiff] * bot0tmp;
							}
						}
					}
				}
			}
		
			const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
			const int bot1index = ((n * bottomheight) + (m-pad_size)) * bottomwidth + (l-pad_size);
			bottom1diff[bot1index + item*bottomcount] = sum / (float)sumelems;
			
		}
	}
	
'''
class _FunctionCorrelation(torch.autograd.Function):

	@staticmethod
	def forward(ctx, first, second, correlation_param_dict):

		in_batch, in_channels, in_height, in_width = first.size()

		pad_size = correlation_param_dict['pad_size']
		kernel_size = correlation_param_dict['kernel_size']
		stride_1 = correlation_param_dict['stride_1']
		stride_2 = correlation_param_dict['stride_2']
		max_displacement = correlation_param_dict['max_displacement']

		padded_height = in_height + 2*pad_size
		padded_width = in_width + 2*pad_size

		kernel_radius = (kernel_size - 1) // 2
		border_size = max_displacement + kernel_radius

		neighborhood_grid_radius = max_displacement // stride_2
		neighborhood_grid_width = neighborhood_grid_radius * 2 + 1

		out_channels = neighborhood_grid_width * neighborhood_grid_width

		out_width = (padded_width - 2 * border_size - 1) // stride_1 + 1
		out_height = (padded_height - 2 * border_size - 1) // stride_1 + 1

		ctx.in_batch = in_batch
		ctx.in_channels = in_channels
		ctx.in_height = in_height
		ctx.in_width = in_width
		ctx.padded_height = padded_height
		ctx.padded_width = padded_width
		ctx.out_width = out_width
		ctx.out_height = out_height
		ctx.out_channels = out_channels
		ctx.kernel_radius = kernel_radius
		ctx.neighborhood_grid_radius = neighborhood_grid_radius
		ctx.neighborhood_grid_width = neighborhood_grid_width
		ctx.border_size = border_size

		ctx.stride_1 = stride_1
		ctx.stride_2 = stride_2
		ctx.max_displacement = max_displacement
		ctx.pad_size = pad_size
		ctx.kernel_size = kernel_size

		rbot0 = first.new_zeros([first.size(0), padded_height, padded_width, first.size(1)])
		rbot1 = first.new_zeros([first.size(0), padded_height, padded_width, first.size(1)])

		ctx.save_for_backward(first, second, rbot0, rbot1)

		assert(first.is_contiguous() == True)
		assert(second.is_contiguous() == True)

		output = first.new_zeros([first.size(0), out_channels, out_height, out_width])

		if first.is_cuda == True:
			n = first.size(2) * first.size(3)
			cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {
				'input': first,
				'output': rbot0
			}))(
				grid=tuple([int((n + 16 - 1) / 16), first.size(1), first.size(0)]),
				block=tuple([16, 1, 1]),
				args=[n, pad_size, first.data_ptr(), rbot0.data_ptr()],
				stream=Stream
			)

			n = second.size(2) * second.size(3)
			cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {
				'input': second,
				'output': rbot1
			}))(
				grid=tuple([int((n + 16 - 1) / 16), second.size(1), second.size(0)]),
				block=tuple([16, 1, 1]),
				args=[n, pad_size, second.data_ptr(), rbot1.data_ptr()],
				stream=Stream
			)

			top_count = output.size(1) * output.size(2) * output.size(3)

			cupy_launch('kernel_Correlation_updateOutput', cupy_kernel('kernel_Correlation_updateOutput', {
				'bottom0': rbot0,
				'bottom1': rbot1,
				'top': output
			}))(
				grid=tuple([output.size(3), output.size(2), output.size(0)]),
				block=tuple([32, 1, 1]),
				shared_mem=kernel_size * kernel_size * first.size(1) * 4,
				args=[in_batch, out_width, out_height, out_channels, top_count,
					  max_displacement, neighborhood_grid_radius,
					  neighborhood_grid_width, kernel_radius, kernel_size,
					  stride_1, stride_2, padded_width, padded_height,
					  in_channels, rbot0.data_ptr(), rbot1.data_ptr(), output.data_ptr()],
				stream=Stream
			)

		elif first.is_cuda == False:
			raise NotImplementedError()

		# end

		return output

	@staticmethod
	def backward(ctx, gradOutput):
		first, second, rbot0, rbot1 = ctx.saved_tensors
#
		in_batch = ctx.in_batch
		in_channels = ctx.in_channels
		in_height = ctx.in_height
		in_width = ctx.in_width
		padded_height = ctx.padded_height
		padded_width = ctx.padded_width
		out_width = ctx.out_width
		out_height = ctx.out_height
		out_channels = ctx.out_channels
		kernel_radius = ctx.kernel_radius
		neighborhood_grid_radius = ctx.neighborhood_grid_radius
		neighborhood_grid_width = ctx.neighborhood_grid_width
		border_size = ctx.border_size

		stride_1 = ctx.stride_1
		stride_2 = ctx.stride_2
		max_displacement = ctx.max_displacement
		pad_size = ctx.pad_size
		kernel_size = ctx.kernel_size

		gradOutput = gradOutput.contiguous()

		assert (gradOutput.is_contiguous() == True)

		gradFirst = first.new_zeros([first.size(0), first.size(1), first.size(2), first.size(3)]) if \
		ctx.needs_input_grad[0] == True else None
		gradSecond = first.new_zeros([first.size(0), first.size(1), first.size(2), first.size(3)]) if \
		ctx.needs_input_grad[1] == True else None

		if first.is_cuda == True:
			if gradFirst is not None:
				bottomcount = first.size(1) * first.size(2) * first.size(3)
				for intSample in range(first.size(0)):
					cupy_launch('kernel_Correlation_updateGradFirst',
								cupy_kernel('kernel_Correlation_updateGradFirst', {
									'bottom0diff': gradFirst,
									'bottom1': rbot1,
									'topdiff': gradOutput
								}))(
						grid=tuple([int((bottomcount + 512 - 1) / 512), 1, 1]),
						block=tuple([512, 1, 1]),
						args=[in_batch, intSample, out_width, out_height, out_channels,
								max_displacement, neighborhood_grid_radius,
								neighborhood_grid_width, kernel_radius, stride_1, stride_2,
								in_width, in_height, padded_width, padded_height,
								in_channels, bottomcount, pad_size,
								gradFirst.data_ptr(), rbot1.data_ptr(), gradOutput.data_ptr()],
						stream=Stream
					)
			# end
			# end

			if gradSecond is not None:
				bottomcount = first.size(1) * first.size(2) * first.size(3)
				for intSample in range(first.size(0)):
					cupy_launch('kernel_Correlation_updateGradSecond',
								cupy_kernel('kernel_Correlation_updateGradSecond', {
									'bottom1diff': gradSecond,
									'bottom0': rbot0,
									'topdiff': gradOutput
								}))(
						grid=tuple([int((bottomcount + 512 - 1) / 512), 1, 1]),
						block=tuple([512, 1, 1]),
						args=[in_batch, intSample, out_width, out_height, out_channels,
								max_displacement, neighborhood_grid_radius,
								neighborhood_grid_width, kernel_radius, stride_1, stride_2,
								in_width, in_height, padded_width, padded_height,
								in_channels, bottomcount, pad_size,
								gradSecond.data_ptr(), rbot0.data_ptr(), gradOutput.data_ptr()],
						stream=Stream
					)
		# end
		# end

		elif first.is_cuda == False:
			raise NotImplementedError()

		# end

		return gradFirst, gradSecond, None


class ModuleCorrelation(nn.Module):
	def __init__(self, pad_size=20, max_displacement=20, stride_1=1, stride_2=2, kernel_size=1):
		super(ModuleCorrelation, self).__init__()

		self.pad_size = pad_size
		self.max_displacement = max_displacement
		self.stride_1 = stride_1
		self.stride_2 = stride_2
		self.kernel_size = kernel_size

		self.correlation_param_dict = {}

		self.correlation_param_dict['pad_size'] = pad_size
		self.correlation_param_dict['kernel_size'] = kernel_size
		self.correlation_param_dict['stride_1'] = stride_1
		self.correlation_param_dict['stride_2'] = stride_2
		self.correlation_param_dict['max_displacement'] = max_displacement

	def forward(self, tensorFirst, tensorSecond):
		return _FunctionCorrelation.apply(tensorFirst, tensorSecond, self.correlation_param_dict)




