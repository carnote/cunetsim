################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/Mob/Mobility.cu \
../src/Mob/MobilityControl.cu \
../src/Mob/ValidityControl.cu 

OBJS += \
./src/Mob/Mobility.o \
./src/Mob/MobilityControl.o \
./src/Mob/ValidityControl.o 


# Each subdirectory must supply rules for building sources it contributes
src/Mob/%.o: ../src/Mob/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVIDIA CUDA Compiler'
	nvcc -I/usr/local/cuda/include -O0 -arch sm_23 -g -c -Xcompiler -fmessage-length=0 -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


