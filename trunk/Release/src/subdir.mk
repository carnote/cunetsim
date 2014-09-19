################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/HelloWorld_kernel.cu \
../src/XXL-NS.cu 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVIDIA CUDA Compiler'
	nvcc -I/usr/local/cuda/include -O3 -c -Xcompiler -fmessage-length=0 -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


