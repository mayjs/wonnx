{%- include "structs.wgsl" -%}

// Input tensor, shape NxCxHxW
@group(0) @binding(0)
var<storage, read> input_tensor: Array;

// Kernel weight tensor, shape CxM/groupxkHxkW
@group(0) @binding(1)
var<storage, read> input_kernel_weights: Array;

{% if i_lens | length == 3 -%}
    @group(0) @binding(2)
    var<storage, read> input_bias: Array;

    @group(0) @binding(3)
    var<storage, read_write> output_0: Array;
{%- else -%}
    @group(0) @binding(2)
    var<storage, read_write> output_0: Array;
{%- endif %}

@compute @workgroup_size({{ workgroup_size_x }}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;

    if (output_idx < {{ o_lens[0] }}u) {
        // Calculate the output coordinates we are responsible for
        let batch = output_idx / {{ o_chunks[0][0] }}u;
        var rest = output_idx % {{ o_chunks[0][0] }}u;

        let channel = rest / {{ o_chunks[0][1] }}u;
        rest = rest % {{ o_chunks[0][1] }}u;

        let y = rest / {{ o_chunks[0][2] }}u;
        let x = rest % {{ o_chunks[0][2] }}u;

        let sample_root_index = batch * {{ i_chunks[0][0] }}u;

        // Calculate the input coordinate range for our output coordinate
        let min_in_y = select(0u, (y - {{ i_shape[1][2] }}u) / {{ stride[0] }}u, y > {{ i_shape[1][2] }}u);
        let max_in_y = select({{ i_shape[0][2] }}u - 1u, y / {{ stride[0] }}u, y / {{ stride[0] }}u < {{ i_shape[0][3] }}u);
        let min_in_x = select(0u, (x - {{ i_shape[1][3] }}u) / {{ stride[1] }}u, x > {{ i_shape[1][3] }}u);
        let max_in_x = select({{ i_shape[0][3] }}u - 1u, x / {{ stride[1] }}u, x / {{ stride[1] }}u < {{ i_shape[0][3] }}u);

        var result: Scalar = Scalar();

        // Now, go over each input channel and apply the corresponing kernel for that channel
        // to calculate the output piece by piece.
        for(var ichannel: u32 = 0u; ichannel < {{ i_shape[0][1] }}u; ichannel = ichannel + 1u) {
            // Base index for the 2D data in the input data
            let base_index = sample_root_index + ichannel * {{ i_chunks[0][1] }}u;
            // Get the starting position of the kernel for the given input and output channel
            let base_kernel_index = ichannel *{{ i_chunks[1][0] }}u + channel * {{ i_chunks[1][1] }}u;

            // Iterate of all potential input values
            for(var in_y: u32 = min_in_y; in_y <= max_in_y; in_y = in_y + 1u) {
                for(var in_x: u32 = min_in_x; in_x <= max_in_x; in_x = in_x + 1u) {
                    let kernel_y = y - (in_y * {{ stride[0] }}u);
                    let kernel_x = x - (in_x * {{ stride[1] }}u);

                    if(kernel_y < {{ i_shape[1][2] }}u && kernel_x < {{ i_shape[1][3] }}u) {
                        result = result + (input_tensor.data[base_index + (in_y * {{ i_chunks[0][2] }}u) + in_x]
                                           * input_kernel_weights.data[base_kernel_index + kernel_y * {{ i_chunks[1][2] }}u + kernel_x]);
                    }
                }
            }
        }
        {% if i_lens | length == 3 -%}
            // Apply Bias if specified
            result = result + input_bias.data[channel];
        {%- endif %}

        output_0.data[output_idx] = result;
    }
}
