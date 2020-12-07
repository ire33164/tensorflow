#ifndef TENSORFLOW_LITE_SIMULATE_NVM_H
#define TENSORFLOW_LITE_SIMULATE_NVM_H

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <cerrno>
#include <cstring>
#include <signal.h>
// #include "/home/chia/Documents/tensorflow/tensorflow/lite/c/common.h"
#include "tensorflow/lite/c/common.h"


#define NVM_SIZE          512 * 1024
#define NODE_IDX          0
/*
#define CONV_BATCH        8  + 100 * 1024
#define CONV_OUT_Y        12 + 100 * 1024
#define CONV_OUT_X        16 + 100 * 1024
#define CONV_OUT_CHANNEL  20 + 100 * 1024
#define CONV_FILTER_X     24 + 100 * 1024
#define CONV_FILTER_Y     28 + 100 * 1024
#define CONV_ACC          32 + 100 * 1024
#define POOLING_BATCH     36 + 100 * 1024
#define POOLING_OUT_Y     40 + 100 * 1024
#define POOLING_OUT_X     44 + 100 * 1024
#define POOLING_CHANNEL   48 + 100 * 1024
#define POOLING_FILTER_X  52 + 100 * 1024
#define POOLING_FILTER_   56 + 100 * 1024
#define POOLING_MAX       60 + 100 * 1024
#define FC_BATCH          61 + 100 * 1024
#define FC_OUT_C          65 + 100 * 1024
#define FC_ACCUM          69 + 100 * 1024
#define FC_D              73 + 100 * 1024
#define VERSION           77 + 100 * 1024
*/

#define OFFSET             50 * 1024
#define NODE_INPUT1       100 * 1024
#define NODE_INPUT2       150 * 1024
#define NODE_OUTPUT1      200 * 1024
#define NODE_OUTPUT2      250 * 1024

#define NODE_INPUT_TEST   300 * 1024
#define NODE_OUTPUT_TEST  350 * 1024

/* data on NVM, made persistent via mmap() with a file */
extern uint8_t *nvm;
extern bool offset_nvm;
extern bool is_power_failure;
extern TfLiteIntermittentParams intermittent_params[2];
extern uint32_t version;

void my_handler(int signum);
void create_mmap();
void my_memcpy(void *dest, const void *src, size_t len);
void read_from_nvm(void *vm_buffer, uint32_t nvm_offset, size_t len);
void write_to_nvm(const void *vm_buffer, uint32_t nvm_offset, size_t len);
void my_erase();
void list_nvm();

#endif // TENSORFLOW_LITE_SIMULATE_NVM_H
