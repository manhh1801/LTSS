#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>



/* Sequential algorithm */
// int main() {
// 	int Size = 6;
// 	double* First = (double*)calloc(Size, sizeof(double));
// 	double* Second = (double*)calloc(Size, sizeof(double));
// 	for(int index = 0; index < Size; index++) {
// 		First[index] = index + 1;
// 		Second[index] = index + 2;
// 	}
//
// 	double* Value = (double*)calloc(Size, sizeof(double));
//
// 	for(int index = 0; index < Size; index++) {
// 		Value[index] = First[index] * Second[index];
// 	}
//
// 	for(int index = 0; index < Size; index++) {
// 		printf("%f ", Value[index]);
// 	}
// }
/*  */



/* Parallel algorithm */
int ProcessID, Processes;

// Master process
double* ParallelProduct_Master(double* FirstOperand, double* SecondOperand, int Size) {
	double* Value = (double*)calloc(Size, sizeof(double));

	// Sending data for slaves
	for(int index = 0; index < Size; index++) {
		int process_id = index % (Processes - 1) + 1;
		int task = index + 1;
		MPI_Send(&task, 1, MPI_INT, process_id, 0, MPI_COMM_WORLD);
		MPI_Send(&FirstOperand[index], 1, MPI_DOUBLE, process_id, 0, MPI_COMM_WORLD);
		MPI_Send(&SecondOperand[index], 1, MPI_DOUBLE, process_id, 0, MPI_COMM_WORLD);
	}
	for(int index = 1; index < Processes; index++) {
		int end = 0;
		MPI_Send(&end, 1, MPI_INT, index, 0, MPI_COMM_WORLD);
	}

	// Receiving and processing data from slaves
	for(int index = 0; index < Size; index++) {
		MPI_Status Status;
		double LocalProduct;
		MPI_Recv(&LocalProduct, 1, MPI_DOUBLE, MPI_ANY_SOURCE, index, MPI_COMM_WORLD, &Status);
		Value[index] = LocalProduct;
	}

	// Terminating
	for(int index = 1; index < Processes; index++) {
		int terminate = -1;
		MPI_Send(&terminate, 1, MPI_INT, index, 0, MPI_COMM_WORLD);
	}

	// Returning value
	return Value;
}

// Slave processes
void ParallelProduct_Slave() {
	while(1) {
		// Receiving status from master
		MPI_Status Status;
		int message;
		MPI_Recv(&message, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &Status);

		// Checking for conditions
		if(message == -1) break;
		if(message == 0) continue;

		// Receiving data from master
		double First, Second;
		MPI_Recv(&First, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &Status);
		MPI_Recv(&Second, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &Status);

		// Processing data and sending back to master
		double LocalProduct = First * Second;
		MPI_Send(&LocalProduct, 1, MPI_DOUBLE, 0, message, MPI_COMM_WORLD);
	}
}

// Entry procedure
int main(int argc, char* argv[])
{
	// Initializing
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcessID);
	MPI_Comm_size(MPI_COMM_WORLD, &Processes);

	// Master process
	if(ProcessID == 0) {
		int Size = 6;
	    double* First = (double*)calloc(Size, sizeof(double));
		double* Second = (double*)calloc(Size, sizeof(double));
	    for(int index = 0; index < Size; index++) {
	        First[index] = index + 1;
	    	Second[index] = index + 2;
	    }

	    double* Product= ParallelProduct_Master(First, Second, Size);
		for(int index = 0; index < Size; index++) {
			printf("%f ", Product[index]);
		}
	}

	// Slave processes
	else {
		ParallelProduct_Slave();
	}



	MPI_Finalize();

	return 0;
}
/*  */