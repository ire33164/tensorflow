# Settings for CCS MSP432

TARGET_ARCH := cortex-m4+fp
CCFLAGS := $(filter-out -std=c11,$(CCFLAGS))
CFLAGS += -std=c11
