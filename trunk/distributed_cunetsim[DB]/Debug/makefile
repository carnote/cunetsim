################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include subdir.mk
-include src/subdir.mk
-include src/Scheduler/subdir.mk
-include src/Mob/subdir.mk
-include src/Init/subdir.mk
-include src/Free/subdir.mk
-include src/Control/subdir.mk
-include src/Connectivity/subdir.mk
-include src/Buffer/subdir.mk
-include src/Topology/subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(C++_DEPS)),)
-include $(C++_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
ifneq ($(strip $(CC_DEPS)),)
-include $(CC_DEPS)
endif
ifneq ($(strip $(CPP_DEPS)),)
-include $(CPP_DEPS)
endif
ifneq ($(strip $(CXX_DEPS)),)
-include $(CXX_DEPS)
endif
ifneq ($(strip $(C_UPPER_DEPS)),)
-include $(C_UPPER_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 
LIBRARIES += \
libUSER_OBJS.so \


# All Target
all: Cunetsim

# Tool invocations
Cunetsim: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: NVCC LinkerC'
	nvcc  -o  "Cunetsim" $(OBJS) $(USER_OBJS) $(LIBS) -lmpi
	@echo 'Finished building target: $@'
	@echo ' '

libUSER_OBJS.so: $(OBJS) $(USER_OBJS)
	@echo 'Invoking: NVCC Linker'
	nvcc -L/usr/local/cuda/lib -shared -o  "libUSER_OBJS.so" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(C++_DEPS)$(OBJS)$(C_DEPS)$(CC_DEPS)$(LIBRARIES)$(CPP_DEPS)$(EXECUTABLES)$(CXX_DEPS)$(C_UPPER_DEPS) Cunetsim
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
