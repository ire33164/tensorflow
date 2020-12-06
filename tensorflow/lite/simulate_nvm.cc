#include "tensorflow/lite/simulate_nvm.h"

uint8_t *nvm;

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
}

void my_memcpy(void *dest, const void *src, size_t len) {
  uint8_t *dest_u = reinterpret_cast<uint8_t *>(dest);
  const uint8_t *src_u = reinterpret_cast<const uint8_t *>(src);
  for(size_t idx = 0; idx < len; ++idx) {
    dest_u[idx] = src_u[idx];
  }
}

void read_from_nvm(void *vm_buffer, uint32_t nvm_offset, size_t len) {
  my_memcpy(vm_buffer, nvm + nvm_offset, len);
}

void write_to_nvm(const void *vm_buffer, uint32_t nvm_offset, size_t len) {
  my_memcpy(nvm + nvm_offset, vm_buffer, len);
}

void my_erase() {
  memset(nvm, 0, NVM_SIZE);
}
