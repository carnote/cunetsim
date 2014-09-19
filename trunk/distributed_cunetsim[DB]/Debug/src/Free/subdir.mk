################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/Free/FreeDevice.cu \
../src/Free/FreeHost.cu 

OBJS += \
./src/Free/FreeDevice.o \
./src/Free/FreeHost.o 


# Each subdirectory must supply rules for building sources it contributes
src/Free/%.o: ../src/Free/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVIDIA CUDA Compiler'
	nvcc -I/usr/local/cuda/include -O0 -arch sm_23 -g -c -Xcompiler -fmessage-length=0 -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


