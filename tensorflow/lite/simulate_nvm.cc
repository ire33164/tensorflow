#include "tensorflow/lite/simulate_nvm.h"

uint8_t *nvm;
uint32_t version = 0;
bool offset_nvm;
bool is_power_failure;
bool is_recovery_mode;
TfLiteIntermittentParams intermittent_params[2];

void my_handler(int signum) {
  printf("-------------------- Temporary INFO ----------------------\n");
  list_nvm();
  printf("-------------------- Power  Failure ----------------------\n");
  exit(signum);
}

void create_mmap() {
  int nvm_fd = -1;
  struct stat stat_buf;
  if (stat("nvm.bin", &stat_buf) != 0) {
    if (errno != ENOENT) {
      perror("Checking nvm.bin failed");
    }
    nvm_fd = open("nvm.bin", O_RDWR|O_CREAT, 0600);
    if(ftruncate(nvm_fd, NVM_SIZE) != 0) {
      if (errno != ENOENT) {
        perror("Ftruncating nvm failed.");
      }
    }
  } else {
    nvm_fd = open("nvm.bin", O_RDWR);
  }
  nvm = reinterpret_cast<uint8_t *>(mmap(NULL, NVM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, nvm_fd, 0));
  if (nvm == MAP_FAILED) {
    perror("mmap() fail.");
  }
  read_from_nvm(&is_recovery_mode, RECOVERY, sizeof(is_recovery_mode));
  close(nvm_fd);
}

void init_nvm() {
  /* resetting nvm */
  my_erase();
  is_recovery_mode = 1;
  write_to_nvm(&is_recovery_mode, RECOVERY, sizeof(is_recovery_mode));
}

void set_params () {
  read_from_nvm(&intermittent_params[0], OFFSET1, sizeof(TfLiteIntermittentParams));
  read_from_nvm(&intermittent_params[1], OFFSET2, sizeof(TfLiteIntermittentParams));

  is_power_failure = intermittent_params[0].version != 0;
  offset_nvm = intermittent_params[0].version >= intermittent_params[1].version ? 0 : 1;
}

void run_finish() {
  is_recovery_mode = 0;
  write_to_nvm(&is_recovery_mode, RECOVERY, sizeof(is_recovery_mode));
}

void my_memcpy(void *dest, const void *src, size_t len) {
  printf("Copying\n");
  uint8_t *dest_u = reinterpret_cast<uint8_t *>(dest);
  const uint8_t *src_u = reinterpret_cast<const uint8_t *>(src);
  for(size_t idx = 0; idx < len; ++idx) {
    dest_u[idx] = src_u[idx];
  }
}

void read_from_nvm(void *vm_buffer, uint32_t nvm_offset, size_t len) {
  my_memcpy(vm_buffer, nvm + nvm_offset, len);
}

void read_from_nvm_segmented(void *vm_buffer, uint32_t nvm_offset, uint16_t total_len, uint16_t segment_size) {
    for (uint16_t idx = 0; idx < total_len; idx += segment_size) {
        read_from_nvm((uint8_t *)vm_buffer + idx, nvm_offset + idx, MIN_VAL(total_len - idx, segment_size));
    }
}

void write_to_nvm(const void *vm_buffer, uint32_t nvm_offset, size_t len) {
  my_memcpy(nvm + nvm_offset, vm_buffer, len);
}

void write_to_nvm_segmented(const void *vm_buffer, uint32_t nvm_offset, uint16_t total_len, uint16_t segment_size) {
    for (uint16_t idx = 0; idx < total_len; idx += segment_size) {
        write_to_nvm((uint8_t *)vm_buffer + idx, nvm_offset + idx, MIN_VAL(total_len - idx, segment_size));
    }
}

void my_erase() {
  memset(nvm, 0, NVM_SIZE);
}

void list_nvm() {
  for (int i = 0; i < 2; ++i) {
    printf("node idx: %ld\n", intermittent_params[i].node_idx);
    printf("OFM_cnt   : %d\n", intermittent_params[i].OFM_cnt);
    printf("version : %d\n", intermittent_params[i].version);
  }
}
