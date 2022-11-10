#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "util.h"

// The width and height of a sudoku board
#define BOARD_DIM 9

// The width and heigh of a square group in a sudoku board
#define GROUP_DIM 3

// The number of boards to pass to the solver at one time
#define BATCH_SIZE 25000

/**
 * A board is an array of 81 cells. Each cell is encoded as a 16-bit integer.
  */
typedef struct board {
  uint16_t cells[BOARD_DIM * BOARD_DIM];
} board_t;

// Declare a few functions. 
void print_board(board_t* board);
__host__ __device__ uint16_t digit_to_cell(int digit);
__host__ __device__ int cell_to_digit(uint16_t cell);

/**
 * This is the kernal to solve the sudoku boards in GPU.
 * more than BATCH_SIZE, but may be less if the total number of input
 *
 *
 * \param boards      An array of boards that should be solved.
 */
__global__ void cell_solver(board_t* boards) {
  size_t cell_idx = threadIdx.x;
  uint16_t current_cell;
  size_t votes;

  // shared memory for all the threads in the block.
  __shared__ board_t board;
  // copy the contents of the board into the shared memory
  board.cells[cell_idx] = boards[blockIdx.x].cells[cell_idx];
  // wait for all the threads to finish copying the boards.
  __syncthreads();

  do {
    current_cell = board.cells[cell_idx];
    if (cell_to_digit(current_cell) != 0) break;
    // loop through the col
    size_t col_idx = cell_idx % 9;
    for (size_t index = col_idx; index < col_idx + 9 * 9; index += 9) {
      if (index == cell_idx) continue;
      int digit_result = cell_to_digit(board.cells[index]);
      if (digit_result != 0) board.cells[cell_idx] &= ~(1 << digit_result);
    }
    if (cell_to_digit(current_cell) != 0) break;
    // loop through the row
    size_t start_idx = cell_idx - col_idx;
    for (size_t index = start_idx; index < start_idx + 9; index++) {
      if (index == cell_idx) continue;
      int digit_result = cell_to_digit(board.cells[index]);
      if (digit_result != 0) board.cells[cell_idx] &= ~(1 << digit_result);
    }
    if (cell_to_digit(current_cell) != 0) break;
    // find the index of the top left corner of the square
    // reduced_index is the index of cell that has the same column
    // index but is in the first row.
    size_t reduced_index = cell_idx - (cell_idx / 27) * 27;
    size_t minor_row = reduced_index / 9;
    size_t minor_col = (reduced_index - minor_row * 9) % 3;
    // start_index is the index of cell at the top left corner that
    // share the same square of the current cell.
    size_t start_index = cell_idx - minor_col - minor_row * 9;
    // loop through the square
    for (size_t row = 0; row < 3; row++) {
      for (size_t col = 0; col < 3; col++) {
        size_t index = start_index + col + row * 9;
        if (index == cell_idx) continue;
        int digit_result = cell_to_digit(board.cells[index]);
        if (digit_result != 0) board.cells[cell_idx] &= ~(1 << digit_result);
      }
    }
    votes = __syncthreads_count(board.cells[cell_idx] != current_cell);

  } while (votes != 0);

  boards[blockIdx.x].cells[cell_idx] = board.cells[cell_idx];
}

/**
 * Take an array of boards and solve them all. 
 *
 * \param boards      An array of boards that should be solved.
 * \param num_boards  The numebr of boards in the boards array
 */
void solve_boards(board_t* cpu_boards, size_t num_boards) {
  // allocate memory in gpu
  board_t* gpu_boards;
  if (cudaMalloc(&gpu_boards, sizeof(board_t) * num_boards) != cudaSuccess) {
    perror("cuda malloc failed.");
    exit(2);
  }
  // copy the content over to gpu
  if (cudaMemcpy(gpu_boards, cpu_boards, sizeof(board_t) * num_boards, cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    perror("cuda memcpy failed. ");
    exit(2);
  }
  // run the kernal over BATCH_SIZE blocks and 81 threads
  cell_solver<<<BATCH_SIZE, 81>>>(gpu_boards);
  // wait for all the threads to finish
  if (cudaDeviceSynchronize() != cudaSuccess) {
    perror("Synchronized failed.");
    exit(2);
  }
  // copy contents from gpu to cpu.
  if (cudaMemcpy(cpu_boards, gpu_boards, sizeof(board_t) * num_boards, cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    perror("cuda memcpy failed. ");
    exit(2);
  }
}

/**
 * Take as input an integer value 0-9 (inclusive) and convert it to the encoded
 * cell form used for solving the sudoku. This encoding uses bits 1-9 to
 * indicate which values may appear in this cell.
 *
 * For example, if bit 3 is set to 1, then the cell may hold a three. Cells that
 * have multiple possible values will have multiple bits set.
 *
 * The input digit 0 is treated specially. This value indicates a blank cell,
 * where any value from one to nine is possible.
 *
 * \param digit   An integer value 0-9 inclusive
 * \returns       The encoded form of digit using bits to indicate which values
 *                may appear in this cell.
 */
__host__ __device__ uint16_t digit_to_cell(int digit) {
  if (digit == 0) {
    // A zero indicates a blank cell. Numbers 1-9 are possible, so set bits 1-9.
    return 0x3FE;
  } else {
    // Otherwise we have a fixed value. Set the corresponding bit in the board.
    return 1 << digit;
  }
}

/*
 * Convert an encoded cell back to its digit form. A cell with two or more
 * possible values will be encoded as a zero. Cells with one possible value
 * will be converted to that value.
 *
 *
 * \param cell  An encoded cell that uses bits to indicate which values could
 *              appear at this point in the board.
 * \returns     The value that must appear in the cell if there is only one
 *              possibility, or zero otherwise.
 */
__host__ __device__ int cell_to_digit(uint16_t cell) {
  // Get the index of the least-significant bit in this cell's value
#if defined(__CUDA_ARCH__)
  int msb = __clz(cell);
  int lsb = sizeof(unsigned int) * 8 - msb - 1;
#else
  int lsb = __builtin_ctz(cell);
#endif

  // Is there only one possible value for this cell? If so, return it.
  // Otherwise return zero.
  if (cell == 1 << lsb)
    return lsb;
  else
    return 0;
}

/**
 * Read in a sudoku board from a string. Boards are represented as an array of
 * 81 16-bit integers. Each integer corresponds to a cell in the board. Bits
 * 1-9 of the integer indicate whether the values 1, 2, ..., 8, or 9 could
 * appear in the given cell. A zero in the input indicates a blank cell, where
 * any value could appear.
 *
 * \param output  The location where the board will be written
 * \param str     The input string that encodes the board
 * \returns       true if parsing succeeds, false otherwise
 */
bool read_board(board_t* output, const char* str) {
  for (int index = 0; index < BOARD_DIM * BOARD_DIM; index++) {
    if (str[index] < '0' || str[index] > '9') return false;

    // Convert the character value to an equivalent integer
    int value = str[index] - '0';

    // Set the value in the board
    output->cells[index] = digit_to_cell(value);
  }

  return true;
}

/**
 * Check through a batch of boards to see how many were solved correctly.
 *
 * \param boards        An array of (hopefully) solved boards
 * \param solutions     An array of solution boards
 * \param num_boards    The number of boards and solutions
 * \param solved_count  Output: A pointer to the count of solved boards.
 * \param error:count   Output: A pointer to the count of incorrect boards.
 */
void check_solutions(board_t* boards,
                     board_t* solutions,
                     size_t num_boards,
                     size_t* solved_count,
                     size_t* error_count) {
  // Loop over all the boards in this batch
  for (int i = 0; i < num_boards; i++) {
    // Does the board match the solution?
    if (memcmp(&boards[i], &solutions[i], sizeof(board_t)) == 0) {
      // Yes. Record a solved board
      (*solved_count)++;
    } else {
      // No. Make sure the board doesn't have any constraints that rule out
      // values that are supposed to appear in the solution.
      bool valid = true;
      for (int j = 0; j < BOARD_DIM * BOARD_DIM; j++) {
        if ((boards[i].cells[j] & solutions[i].cells[j]) == 0) {
          valid = false;
        }
      }

      // If the board contains an incorrect constraint, record an error
      if (!valid) (*error_count)++;
    }
  }
}

/**
 * Entry point for the program
 */
int main(int argc, char** argv) {
  // Check arguments
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <input file name>\n", argv[0]);
    exit(1);
  }

  // Try to open the input file
  FILE* input = fopen(argv[1], "r");
  if (input == NULL) {
    fprintf(stderr, "Failed to open input file %s.\n", argv[1]);
    perror(NULL);
    exit(2);
  }

  // Keep track of total boards, boards solved, and incorrect outputs
  size_t board_count = 0;
  size_t solved_count = 0;
  size_t error_count = 0;

  // Keep track of time spent solving
  size_t solving_time = 0;

  // Reserve space for a batch of boards and solutions
  board_t boards[BATCH_SIZE];
  board_t solutions[BATCH_SIZE];

  // Keep track of how many boards we've read in this batch
  size_t batch_count = 0;

  // Read the input file line-by-line
  char* line = NULL;
  size_t line_capacity = 0;
  while (getline(&line, &line_capacity, input) > 0) {
    // Read in the starting board
    if (!read_board(&boards[batch_count], line)) {
      fprintf(stderr, "Skipping invalid board...\n");
      continue;
    }

    // Read in the solution board
    if (!read_board(&solutions[batch_count], line + BOARD_DIM * BOARD_DIM + 1)) {
      fprintf(stderr, "Skipping invalid board...\n");
      continue;
    }

    // Move to the next index in the batch
    batch_count++;

    // Also increment the total count of boards
    board_count++;

    // If we finished a batch, run the solver
    if (batch_count == BATCH_SIZE) {
      size_t start_time = time_ms();
      solve_boards(boards, batch_count);
      solving_time += time_ms() - start_time;

      check_solutions(boards, solutions, batch_count, &solved_count, &error_count);

      // Reset the batch count
      batch_count = 0;
    }
  }

  // Check if there's an incomplete batch to solve
  if (batch_count > 0) {
    size_t start_time = time_ms();
    solve_boards(boards, batch_count);
    solving_time += time_ms() - start_time;

    check_solutions(boards, solutions, batch_count, &solved_count, &error_count);
  }

  // Print stats
  double seconds = (double)solving_time / 1000;
  double solving_rate = (double)solved_count / seconds;

  // Don't print nan when solver is not implemented
  if (seconds < 0.01) solving_rate = 0;

  printf("Boards: %lu\n", board_count);
  printf("Boards Solved: %lu\n", solved_count);
  printf("Errors: %lu\n", error_count);
  printf("Total Solving Time: %lums\n", solving_time);
  printf("Solving Rate: %.2f sudoku/second\n", solving_rate);

  return 0;
}
