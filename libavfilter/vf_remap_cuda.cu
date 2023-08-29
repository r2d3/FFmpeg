extern "C" {

__global__ void Remap_Cuda(
    int x_position, int y_position,
    const unsigned char* top, int top_linesize,
    const unsigned char* bottom, int bottom_linesize,
    int top_w, int top_h,
    const unsigned short* xmap, int xmap_linesize,
    const unsigned short* ymap, int ymap_linesize,
    int map_w, int map_h,
    unsigned char* out, int out_linesize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < 0 || x >= map_w || y < 0 || y >= map_h)
        return;
        
    int src_x = xmap[x + y * xmap_linesize];
    int src_y = ymap[x + y * ymap_linesize];
    
    unsigned char v;
    if (src_y >= 2*top_h || src_x >= top_w)
        v = 0;
    else if (src_y < top_h) // source in top
        v = top[src_x + src_y * top_linesize];
    else
        v = bottom[src_x + (src_y - top_h) * bottom_linesize];
        
    out[x + y * out_linesize] = v;
}

}

