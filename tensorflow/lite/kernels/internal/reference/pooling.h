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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_POOLING_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_POOLING_H_

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/simulate_nvm.h"

namespace tflite {
namespace reference_ops {

inline void AveragePool(const PoolParams& params,
                        const RuntimeShape& input_shape,
                        const float* input_data,
                        const RuntimeShape& output_shape, float* output_data) {
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          float total = 0.f;
          float filter_count = 0;
          for (int filter_y = filter_y_start; filter_y < filter_y_end;
               ++filter_y) {
            for (int filter_x = filter_x_start; filter_x < filter_x_end;
                 ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              total +=
                  input_data[Offset(input_shape, batch, in_y, in_x, channel)];
              filter_count++;
            }
          }
          const float average = total / filter_count;
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] =
              ActivationFunctionWithMinMax(average, params.float_activation_min,
                                           params.float_activation_max);
        }
      }
    }
  }
}

inline void AveragePool(const PoolParams& params,
                        const RuntimeShape& input_shape,
                        const uint8_t* input_data,
                        const RuntimeShape& output_shape,
                        uint8_t* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          int32_t acc = 0;
          int filter_count = 0;
          for (int filter_y = filter_y_start; filter_y < filter_y_end;
               ++filter_y) {
            for (int filter_x = filter_x_start; filter_x < filter_x_end;
                 ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              acc +=
                  input_data[Offset(input_shape, batch, in_y, in_x, channel)];
              filter_count++;
            }
          }
          acc = (acc + filter_count / 2) / filter_count;
          acc = std::max(acc, params.quantized_activation_min);
          acc = std::min(acc, params.quantized_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] =
              static_cast<uint8_t>(acc);
        }
      }
    }
  }
}

inline void L2Pool(const PoolParams& params, const RuntimeShape& input_shape,
                   const float* input_data, const RuntimeShape& output_shape,
                   float* output_data) {
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          float sum_squares = 0.f;
          int filter_count = 0;
          for (int filter_y = filter_y_start; filter_y < filter_y_end;
               ++filter_y) {
            for (int filter_x = filter_x_start; filter_x < filter_x_end;
                 ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              const float val =
                  input_data[Offset(input_shape, batch, in_y, in_x, channel)];
              sum_squares += val * val;
              filter_count++;
            }
          }
          const float l2pool_result = std::sqrt(sum_squares / filter_count);
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] =
              ActivationFunctionWithMinMax(l2pool_result,
                                           params.float_activation_min,
                                           params.float_activation_max);
        }
      }
    }
  }
}

inline void MaxPool(const PoolParams& params, const RuntimeShape& input_shape,
                    const float* input_data, const RuntimeShape& output_shape,
                    float* output_data) {
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          float max = std::numeric_limits<float>::lowest();
          for (int filter_y = filter_y_start; filter_y < filter_y_end;
               ++filter_y) {
            for (int filter_x = filter_x_start; filter_x < filter_x_end;
                 ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              max = std::max(
                  max,
                  input_data[Offset(input_shape, batch, in_y, in_x, channel)]);
            }
          }
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] =
              ActivationFunctionWithMinMax(max, params.float_activation_min,
                                           params.float_activation_max);
        }
      }
    }
  }
}

inline void MaxPool(const PoolParams& params, const RuntimeShape& input_shape,
                    const uint8_t* input_data, const RuntimeShape& output_shape,
                    uint8_t* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_GE(params.quantized_activation_min, 0);
  TFLITE_DCHECK_LE(params.quantized_activation_max, 255);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  // for intermittent
  const int input_length = input_height * input_width * depth;
  const int output_length = output_height * output_width * depth;

  if (!is_power_failure) {
    // Backing up entire input data to NVM in order to avoid lose input data.
    // Two cases:
    // 1. When the program is running first, there is no the other version.
    // 2. When the program is not running first, the other version exists.
    intermittent_params[offset_nvm].input_version = !intermittent_params[offset_nvm].input_version;
    write_to_nvm(const_cast<uint8_t *>(input_data), intermittent_params[offset_nvm].input_version ? NODE_INPUT2 : NODE_INPUT1, input_length);
    /*
    printf("Length %d\n", input_length);
    printf("Finish writing input to NVM\n");
    printf ("Input data: ");
    for (int i = 0; i < input_length; ++i)
      printf(" %d", input_data[i]);
    printf("\n");
    */
  } else {
    // Recover the node's input and output in VM.
    read_from_nvm(const_cast<uint8_t *>(input_data), intermittent_params[offset_nvm].input_version ? NODE_INPUT2 : NODE_INPUT1, input_length);
    /*
    printf("Finish reading input data from NVM\n");
    printf ("Input data: ");
    for (int i = 0; i < input_length; ++i)
      printf(" %d", input_data[i]);
    printf("\n");
    */
    read_from_nvm(reinterpret_cast<void *>(output_data), offset_nvm ? NODE_OUTPUT2 : NODE_OUTPUT1, output_length);
  }

  size_t node_idx;
  bool input_version;
  node_idx = intermittent_params[offset_nvm].node_idx;
  input_version = intermittent_params[offset_nvm].input_version;
  for (int batch = 0; batch < batches; ++batch) {
    if (is_power_failure) batch = intermittent_params[offset_nvm].batch;
    for (int out_y = 0; out_y < output_height; ++out_y) {
      if (is_power_failure) out_y = intermittent_params[offset_nvm].out_y;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        if (is_power_failure) out_x = intermittent_params[offset_nvm].out_x;
        for (int channel = 0; channel < depth; ++channel) {
          if (is_power_failure) {
            version = intermittent_params[offset_nvm].version + 1;
            channel = intermittent_params[offset_nvm].out_channel + 1;
            is_power_failure = false;
            offset_nvm = !offset_nvm;
          }
          // Finish the operator so we do not need to execute again
          if (channel >= depth) continue;


          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          uint8_t max = 0;
          for (int filter_y = filter_y_start; filter_y < filter_y_end;
               ++filter_y) {
            for (int filter_x = filter_x_start; filter_x < filter_x_end;
                 ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              max = std::max(
                  max,
                  input_data[Offset(input_shape, batch, in_y, in_x, channel)]);
            }
          }
          max = std::max<uint8_t>(max, params.quantized_activation_min);
          max = std::min<uint8_t>(max, params.quantized_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] =
              static_cast<uint8_t>(max);
          // Backing up output into NVM.
          write_to_nvm(reinterpret_cast<void *>(output_data), offset_nvm ? NODE_OUTPUT2 : NODE_OUTPUT1, output_length);
          // Checkpoint forward progress infomation
          intermittent_params[offset_nvm].node_idx = node_idx;
          intermittent_params[offset_nvm].input_version = input_version;
          intermittent_params[offset_nvm].batch = batch;
          intermittent_params[offset_nvm].out_y = out_y;
          intermittent_params[offset_nvm].out_x = out_x;
          intermittent_params[offset_nvm].out_channel = channel;
          intermittent_params[offset_nvm].version = version;
          write_to_nvm(&intermittent_params[offset_nvm], offset_nvm ? OFFSET : 0, sizeof(TfLiteIntermittentParams));
          // printf("version %d\n", version);
          ++version;
          offset_nvm = !offset_nvm;
        }
      }
    }
  }
}
}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_POOLING_H_
