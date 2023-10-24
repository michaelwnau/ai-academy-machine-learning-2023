# Define variables
CC = gcc
CFLAGS = -Wall -Werror
SRC_DIR = src
BUILD_DIR = build
TARGET = my_program

# Define source files
SRCS = $(wildcard $(SRC_DIR)/*.c)

# Define object files
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# Define phony targets
.PHONY: all clean

# Default target
all: $(BUILD_DIR)/$(TARGET)

# Build target
$(BUILD_DIR)/$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@

# Build object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Create build directory
$(BUILD_DIR):
	mkdir -p $@

# Clean target
clean:
	rm -rf $(BUILD_DIR)
