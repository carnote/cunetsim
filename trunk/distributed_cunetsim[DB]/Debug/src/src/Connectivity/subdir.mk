################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/Connectivity/Connectivity.cu \
../src/Connectivity/ConnectivityControl.cu 

OBJS += \
./src/Connectivity/Connectivity.o \
./src/Connectivity/ConnectivityControl.o 


# Each subdirectory must supply rules for building sources it contributes
src/Connectivity/%.o: ../src/Connectivity/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVIDIA CUDA Compiler'
	nvcc -I/usr/local/cuda/include -O0 -arch sm_23 -g -c -Xcompiler -fmessage-length=0 -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


