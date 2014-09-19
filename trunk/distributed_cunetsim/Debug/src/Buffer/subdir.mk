################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/Buffer/Buffer.cu \
../src/Buffer/Receiver.cu \
../src/Buffer/Sender.cu 

OBJS += \
./src/Buffer/Buffer.o \
./src/Buffer/Receiver.o \
./src/Buffer/Sender.o 


# Each subdirectory must supply rules for building sources it contributes
src/Buffer/%.o: ../src/Buffer/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVIDIA CUDA Compiler'
	nvcc -I/usr/local/cuda/include -O0 -arch sm_23 -g -c -Xcompiler -fmessage-length=0 -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


