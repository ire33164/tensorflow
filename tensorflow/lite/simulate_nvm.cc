#include "/home/chia/Documents/tensorflow/tensorflow/lite/simulate_nvm.h"
// #include "tensorflow/lite/simulate_nvm.h"

uint8_t *nvm;
bool offset_nvm;
bool is_power_failure;
TfLiteIntermittentParams intermittent_params[2];

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
  read_from_nvm(&intermittent_params[0], 0, sizeof(TfLiteIntermittentParams));
  read_from_nvm(&intermittent_params[1], OFFSET, sizeof(TfLiteIntermittentParams));
  /*
  uint32_t version1, version2;
  read_from_nvm(&version1, VERSION, sizeof(version1));
  read_from_nvm(&version2, VERSION + OFFSET, sizeof(version2));
  */
  is_power_failure = intermittent_params[0].version != 0;
  offset_nvm = intermittent_params[0].version >= intermittent_params[1].version ? 0 : 1;
  close(nvm_fd);
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

void list_nvm() {
  // read_from_nvm(&intermittent_params[0], 0, sizeof(TfLiteIntermittentParams));
  // read_from_nvm(&intermittent_params[1], OFFSET, sizeof(TfLiteIntermittentParams));
  printf("first address %p\n", &intermittent_params[0]);
  printf("second address %p\n", &intermittent_params[1]);
  printf("size %ld\n", sizeof(TfLiteIntermittentParams));
  for (int i = 0; i < 2; ++i) {
    printf("node idx: %ld\n", intermittent_params[i].node_idx);
    printf("batch   : %d\n", intermittent_params[i].batch);
    printf("out y   : %d\n", intermittent_params[i].out_y);
    printf("out x   : %d\n", intermittent_params[i].out_x);
    printf("out c   : %d\n", intermittent_params[i].out_channel);
    printf("version : %d\n", intermittent_params[i].version);
  }
  /*
  read_from_nvm(&node_idx, NODE_IDX + offset_nvm * OFFSET, sizeof(node_idx));
  read_from_nvm(&batch, CONV_BATCH + offset_nvm * OFFSET, sizeof(batch));
  read_from_nvm(&out_y, CONV_OUT_Y + offset_nvm * OFFSET, sizeof(out_y));
  read_from_nvm(&out_x, CONV_OUT_X + offset_nvm * OFFSET, sizeof(out_x));
  read_from_nvm(&out_channel, CONV_OUT_CHANNEL + offset_nvm * OFFSET, sizeof(out_channel));
  read_from_nvm(&version, VERSION + offset_nvm * OFFSET, sizeof(version));
  printf("node idx: %ld\n", node_idx);
  printf("batch   : %d\n", batch);
  printf("out y   : %d\n", out_y);
  printf("out x   : %d\n", out_x);
  printf("out c   : %d\n", out_channel);
  printf("version : %d\n", version);
  */
}
