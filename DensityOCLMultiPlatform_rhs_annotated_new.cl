void kernel __attribute__((reqd_work_group_size(128, 1, 1))) cscheme(global const int* starting_points,
global const float* data_points,global float* C, private int startid) {
    float grid_index[5];
    float grid_level_2[5];
    if (get_global_id(0) < 61183) {
        for (int d = 0; d < 5; d++) {
            grid_index[d] = (float)(starting_points[(startid + get_global_id(0)) * 2 * 5 + 2 * d]);
            grid_level_2[d] =
                (float)(1 << starting_points[(startid + get_global_id(0)) * 2 * 5 + 2 * d + 1]);
        }

    } else {
        for (int d = 0; d < 10; d++) {
            grid_index[d] = 1.0;
            grid_level_2[d] = 2.0;
        }
    }
    local float data_group[128 * 5];

    float result = 0.0f;
    for (int outer_data_index = 0; outer_data_index < 371908;
            outer_data_index += get_local_size(0)) {
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int d = 0; d < 5; d++) {
            int data_index = outer_data_index + get_local_id(0);
            if (data_index < 371908) {
                data_group[get_local_id(0) * 5 + d] = data_points[data_index * 5 + d];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int inner_data_index = 0; inner_data_index < get_local_size(0); inner_data_index += 1) { // * grid_size
            int data_index = outer_data_index + inner_data_index;
            if (data_index >= 371908) {
                break;
            }
            float eval = 1.0f;
            for (int d = 0; d < 5; d++) { // * d
                float eval_1d = grid_level_2[d];
                eval_1d *= data_group[inner_data_index * 5 + d]; // 1
                eval_1d -= grid_index[d]; // 1
                eval_1d = fabs(eval_1d); // 1
                eval_1d = 1 - eval_1d; // 1
                if (eval_1d < 0) eval_1d = 0; // 1 (== max())
                eval *= eval_1d; // 1
            }
            result += eval; // 1
            // sum: 6 * d + 1
        }
    }
    result /= 371908.0f;
    if (get_global_id(0) < 61183) {
        C[get_global_id(0)] = result;
    }
}
