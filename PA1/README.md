# PA0: Add Two Matrices
Program prompts user for dimension of square matrix. Also prompts for striding or not.
To change grid, block layout, there is a switch statement within the add.cu functions that select thread ID based on grid, block layout.

Program prints out time it took to add matrices sequentially and on the device.
