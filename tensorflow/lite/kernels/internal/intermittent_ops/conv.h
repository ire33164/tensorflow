/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_INTERMITTENT_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_INTERMITTENT_CONV_H_

#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/simulate_nvm.h"

namespace tflite {

namespace intermittent_ops {


inline void Conv(const ConvParams& params, const RuntimeShape& input_shape,
                 const float* input_data, const RuntimeShape& filter_shape,
                 const float* filter_data, const RuntimeShape& bias_shape,
                 const float* bias_data, const RuntimeShape& output_shape,
                 float* output_data, const RuntimeShape& im2col_shape,
                 float* im2col_data) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  (void)im2col_data;   // only used in optimized code.
  (void)im2col_shape;  // only used in optimized code.
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          float total = 0.f;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                float input_value = input_data[Offset(input_shape, batch, in_y,
                                                      in_x, in_channel)];
                float filter_value = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                total += (input_value * filter_value);
              }
            }
          }
          float bias_value = 0.0f;
          if (bias_data) {
            bias_value = bias_data[out_channel];
          }
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              ActivationFunctionWithMinMax(total + bias_value,
                                           output_activation_min,
                                           output_activation_max);
        }
      }
    }
  }
}

inline void Conv(const ConvParams& params, const RuntimeShape& input_shape,
                 const uint8_t* input_data, const RuntimeShape& filter_shape,
                 const uint8_t* filter_data, const RuntimeShape& bias_shape,
                 const int32_t* bias_data, const RuntimeShape& output_shape,
                 uint8_t* output_data, const RuntimeShape& im2col_shape,
                 uint8_t* im2col_data, void* cpu_backend_context) {
  printf("Intermittent CONV \n");
  (void)cpu_backend_context;  // only used in optimized code.
  (void)im2col_data;   // only used in optimized code.
  (void)im2col_shape;  // only used in optimized code.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int input_length = input_height * input_width * input_depth;
  const int output_length = output_height * output_width * output_depth;
  int tmp_batch = 0;
  int tmp_out_y = 0;
  int tmp_out_x = 0;
  int tmp_out_channel = 0;

  if (!is_power_failure) {
    // Backing up entire input data to NVM in order to avoid lose input data.
    // Two cases:
    // 1. When the program is running first, there is no the other version.
    // 2. When the program is not running first, the other version exists.
    intermittent_params[offset_nvm].input_version = !intermittent_params[offset_nvm].input_version;
    write_to_nvm(const_cast<uint8_t *>(input_data), intermittent_params[offset_nvm].input_version ? NODE_INPUT2 : NODE_INPUT1, input_length);
  } else {
    // Recover the node's input and output in VM.
    read_from_nvm(const_cast<uint8_t *>(input_data), intermittent_params[offset_nvm].input_version ? NODE_INPUT2 : NODE_INPUT1, input_length);
    read_from_nvm(reinterpret_cast<void *>(output_data), offset_nvm ? NODE_OUTPUT2 : NODE_OUTPUT1, output_length);
    int OFM_cnt = intermittent_params[offset_nvm].OFM_cnt;
    tmp_batch = OFM_cnt / (output_height * output_width * output_depth);
    tmp_out_y =(OFM_cnt - tmp_batch * (output_height * output_width * output_depth)) / (output_width * output_depth) ;
    tmp_out_x = (OFM_cnt - tmp_batch * (output_height * output_width * output_depth) - tmp_out_y * (output_width * output_depth)) / output_depth;
    tmp_out_channel = OFM_cnt - tmp_batch * (output_height * output_width * output_depth) - tmp_out_y * (output_width * output_depth) - tmp_out_x * output_depth;
    /*
    printf("Calculate\n");
    printf("batch: %d\nout_y: %d\nout_x: %d\nout_channel: %d\n", tmp_batch, tmp_out_y, tmp_out_x, tmp_out_channel);
    printf("Correct\n");
    printf("batch: %d\nout_y: %d\nout_x: %d\nout_channel: %d\n", intermittent_params[offset_nvm].batch, intermittent_params[offset_nvm].out_y, intermittent_params[offset_nvm].out_x, intermittent_params[offset_nvm].out_channel);
    */
  }

  size_t node_idx;
  bool input_version;
  node_idx = intermittent_params[offset_nvm].node_idx;
  input_version = intermittent_params[offset_nvm].input_version;
  for (int batch = tmp_batch; batch < batches; ++batch) {
    if (is_power_failure) batch = tmp_batch;
    for (int out_y = 0; out_y < output_height; ++out_y) {
      if (is_power_failure) out_y = tmp_out_y;
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        if (is_power_failure) out_x = tmp_out_x;
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          if (is_power_failure) {
            version = intermittent_params[offset_nvm].version + 1;
            out_channel = tmp_out_channel + 1;
            is_power_failure = false;
            offset_nvm = !offset_nvm;
          }
          // Finish the operator so we do not need to execute again
          if (out_channel >= output_depth) continue;

          int32_t acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                int32_t input_val = input_data[Offset(input_shape, batch, in_y,
                                                      in_x, in_channel)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                acc +=
                    (filter_val + filter_offset) * (input_val + input_offset);
              }
            }
          }
          if (bias_data) {
            acc += bias_data[out_channel];
          }
          acc = MultiplyByQuantizedMultiplier(acc, output_multiplier,
                                              output_shift);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<uint8_t>(acc);
          // Backing up output into NVM.
          write_to_nvm(reinterpret_cast<void *>(output_data), offset_nvm ? NODE_OUTPUT2 : NODE_OUTPUT1, output_length);
          // Checkpoint forward progress infomation
          intermittent_params[offset_nvm].node_idx = node_idx;
          intermittent_params[offset_nvm].input_version = input_version;
          intermittent_params[offset_nvm].OFM_cnt = batch * output_height * output_width * output_depth + out_y * output_width * output_depth + out_x * output_depth + out_channel;
          /*
          intermittent_params[offset_nvm].batch = batch;
          intermittent_params[offset_nvm].out_y = out_y;
          intermittent_params[offset_nvm].out_x = out_x;
          intermittent_params[offset_nvm].out_channel = out_channel;
          */
          intermittent_params[offset_nvm].version = version;
          write_to_nvm(&intermittent_params[offset_nvm], offset_nvm ? OFFSET : 0, sizeof(TfLiteIntermittentParams));
          // printf("version %d\n", version);
          ++version;
          offset_nvm = !offset_nvm;
        }
      }
    }
  }
  /*
  printf("-----------------------------------------\n");
  list_nvm();
  printf("-----------------------------------------\n");
  */
}

inline void HybridConvPerChannel(
    const ConvParams& params, float* scaling_factors_ptr,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& filter_shape, const int8_t* filter_data,
    const RuntimeShape& bias_shape, const float* bias_data,
    const RuntimeShape& output_shape, float* output_data,
    const RuntimeShape& im2col_shape, int8_t* im2col_data,
    const float* per_channel_scale, int32_t* input_offset) {
  (void)im2col_data;   // only used in optimized code.
  (void)im2col_shape;  // only used in optimized code.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          int32_t acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height)) {
                  int32_t input_val = input_data[Offset(
                      input_shape, batch, in_y, in_x, in_channel)];
                  int32_t filter_val =
                      filter_data[Offset(filter_shape, out_channel, filter_y,
                                         filter_x, in_channel)];
                  acc += filter_val * (input_val - input_offset[batch]);
                }
              }
            }
          }
          float acc_float =
              acc * per_channel_scale[out_channel] * scaling_factors_ptr[batch];
          if (bias_data) {
            acc_float += bias_data[out_channel];
          }
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              ActivationFunctionWithMinMax(acc_float, output_activation_min,
                                           output_activation_max);
        }
      }
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CONV_H_
