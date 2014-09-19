################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/Init/DeviceMemoryAlloction.cu \
../src/Init/HostMemoryAllocation.cu \
../src/Init/Init.cu \
../src/Init/InitCell.cu \
../src/Init/InitCellGen.cu \
../src/Init/InitCellGen2D.cu \
../src/Init/InitGeo.cu \
../src/Init/InitNode.cu \
../src/Init/InitRouterBuffer.cu 

OBJS += \
./src/Init/DeviceMemoryAlloction.o \
./src/Init/HostMemoryAllocation.o \
./src/Init/Init.o \
./src/Init/InitCell.o \
./src/Init/InitCellGen.o \
./src/Init/InitCellGen2D.o \
./src/Init/InitGeo.o \
./src/Init/InitNode.o \
./src/Init/InitRouterBuffer.o 


# Each subdirectory must supply rules for building sources it contributes
src/Init/%.o: ../src/Init/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVIDIA CUDA Compiler'
	nvcc -I/usr/local/cuda/include -O0 -arch sm_23 -g -c -Xcompiler -fmessage-length=0 -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


