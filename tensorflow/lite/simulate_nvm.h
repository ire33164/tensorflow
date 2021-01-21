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
#include "tensorflow/lite/c/common.h"


#define NVM_SIZE          512 * 1024
/* 0 ~ 15 for test SPI */
#define OFFSET1                   16
#define OFFSET2                   64
#define RECOVERY                 112
/* changed accroding model and FRAM size */
#define NODE_INPUT1       100 * 1024
#define NODE_INPUT2       150 * 1024
#define NODE_OUTPUT1      200 * 1024
#define NODE_OUTPUT2      250 * 1024

#define MAX_ACCESS_LENGTH 2 * 1024
#define MIN_VAL(x, y) ((x) < (y) ? (x) : (y))


/* data on NVM, made persistent via mmap() with a file */
extern uint8_t *nvm;
extern bool offset_nvm;
extern bool is_power_failure;
extern bool is_recovery_mode;
extern TfLiteIntermittentParams intermittent_params[2];
extern uint32_t version;

void my_handler(int signum);
void create_mmap();
void init_nvm();
void set_params();
void my_memcpy(void *dest, const void *src, size_t len);
void read_from_nvm(void *vm_buffer, uint32_t nvm_offset, size_t len);
void read_from_nvm_segmented(void *vm_buffer, uint32_t nvm_offset, uint16_t total_len, uint16_t segment_size);
void write_to_nvm(const void *vm_buffer, uint32_t nvm_offset, size_t len);
void write_to_nvm_segmented(const void *vm_buffer, uint32_t nvm_offset, uint16_t total_len, uint16_t segment_size);
void my_erase();
void list_nvm();
void run_finish();

#endif // TENSORFLOW_LITE_SIMULATE_NVM_H
