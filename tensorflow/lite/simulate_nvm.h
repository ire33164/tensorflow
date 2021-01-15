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
#define OFFSET             25 * 1024
#define RECOVERY           50 * 1024
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
extern bool is_recovery_mode;
extern TfLiteIntermittentParams intermittent_params[2];
extern uint32_t version;

void my_handler(int signum);
void create_mmap();
void init_nvm();
void set_params();
void my_memcpy(void *dest, const void *src, size_t len);
void read_from_nvm(void *vm_buffer, uint32_t nvm_offset, size_t len);
void write_to_nvm(const void *vm_buffer, uint32_t nvm_offset, size_t len);
void my_erase();
void list_nvm();
void run_finish();

#endif // TENSORFLOW_LITE_SIMULATE_NVM_H
