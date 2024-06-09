#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../lib/mpi.h"
// #include <mpi.h>

/* Process information */
int ProcessID, Processes;
/*  */

/* Tasks */
typedef struct {
    int Start;
    int End;
} TaskAssignment;
/*  */

/* Data */
typedef struct {
    double *Input;
    double Output;
} Data;
/*  */

/* Parsing input from file */
Data* parseFile(int* Size, int Features, char* FilePath) {
    /* Initializing*/
    Data* DataSet = calloc(1024, sizeof(Data));
    *Size = 0;

    /* Processing */
    FILE* File = fopen(FilePath, "r");
    while(!feof(File)) {
        Data DataPoint;
        char* buffer = calloc(1024, sizeof(char));
        fgets(buffer, 1024, File);
        char* value = strtok(buffer, ",");
        if(value == NULL) { break; }
        DataPoint.Output = atof(value);
        DataPoint.Input = (double*)calloc(Features, sizeof(double));
        DataPoint.Input[0] = 1;
        for(int index = 1; index < Features; index++) {
            value = strtok(NULL, ",");
            if(value == NULL) { break; }
            DataPoint.Input[index] = atof(value);
        }
        DataSet[*Size] = DataPoint;
        *Size += 1;
    }
    fclose(File);

    /* Returning value */
    return DataSet;
}
/*  */

/* Entry procedure */
int main(int argc, char** argv) {
    /* Initializing parallel environment */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcessID);
    MPI_Comm_size(MPI_COMM_WORLD, &Processes);

    /* Master */
    if(ProcessID == 0) {
        /* Validating terminal arguments */


        /* Initializing data  */
        Data* DataSet = NULL;
        int Size = 0;
        int Features = atoi(argv[1]) + 1;
        DataSet = (Data*)parseFile(&Size, Features, argv[2]);

        /* Assigning tasks */
        TaskAssignment Tasks[Processes];
        int quotient = Size / Processes; int remainder = Size % Processes;
        for(int process = 0; process < Processes; process++) {
            Tasks[process].Start = quotient * process + (process < remainder ? process : remainder);
            Tasks[process].End = Tasks[process].Start + (process < remainder ? quotient : quotient - 1);
            int TaskCount = Tasks[process].End - Tasks[process].Start + 1;
            MPI_Send(&TaskCount, 1, MPI_INT, process, 0, MPI_COMM_WORLD);
        }
        int TaskCount = Tasks[0].End - Tasks[0].Start + 1;
        if(Size < Processes) { Processes = Size; }

        /* Sending data to slaves */
        for(int process = 1; process < Processes; process++) {
            for(int task = Tasks[process].Start; task <= Tasks[process].End; task++) {
                MPI_Send(DataSet[task].Input, Features, MPI_DOUBLE, process, 0, MPI_COMM_WORLD);
                MPI_Send(&DataSet[task].Output, 1, MPI_DOUBLE, process, 0, MPI_COMM_WORLD);
            }
        }

        // /* Gradient Descent */
        // double LearningRate = 0.0001;
        // double AcceptedError = 0.01;
        // double* Parameters = calloc(Features, sizeof(double));
        // double* Derivatives = calloc(Features, sizeof(double));
        // for(int feature = 0; feature < Features; feature++) {
        //     Parameters[feature] = 1;
        //     Derivatives[feature] = 1;
        // }
        // int loop = 0;
        // while(1) {
        //     /* Checking for loop exit */
        //     int exit = 1;
        //     for(int feature = 0; feature < Features; feature++) { if(fabs(Derivatives[feature]) > AcceptedError) { exit = 0; } }
        //     exit = exit == 1 || loop == 1000000 ? 1 : 0;
        //     for(int process = 1; process < Processes; process++) { MPI_Send(&exit, 1, MPI_INT, process, 0, MPI_COMM_WORLD); }
        //     if(exit == 1) { break; }
        //
        //     /* Sending current parameters to slave */
        //     for(int process = 1; process < Processes; process++) { MPI_Send(Parameters, Features, MPI_DOUBLE, process, 0, MPI_COMM_WORLD); }
        //
        //     /* Doing the calculation for the assigned tasks */
        //     double* Error = calloc(Features, sizeof(double));
        //     for(int task = 0; task < TaskCount; task++) {
        //         for(int feature = 0; feature < Features; feature++) {
        //             Error[task] += DataSet[task].Input[feature] * Parameters[feature];
        //         }
        //         Error[task] -= DataSet[task].Output;
        //     }
        //
        //     for(int feature = 0; feature < Features; feature++) {
        //         Derivatives[feature] = 0;
        //         for(int task = 0; task < TaskCount; task++) { Derivatives[feature] += DataSet[task].Input[feature] * Error[task]; }
        //     }
        //
        //     /* Receiving partial derivatives back from slaves and calculating derivatives */
        //     for(int process = 1; process < Processes; process++) {
        //         double* PartialDerivatives = calloc(Features, sizeof(double));
        //         MPI_Recv(PartialDerivatives, Features, MPI_DOUBLE, process, 0, MPI_COMM_WORLD, NULL);
        //         for(int feature = 0; feature < Features; feature++) { Derivatives[feature] -= PartialDerivatives[feature]; }
        //     }
        //
        //     /* Updating parameters */
        //     for(int feature = 0; feature < Features + 1; feature++) { Parameters[feature] -= LearningRate * Derivatives[feature]; }
        //
        //     /* Updating loop count */
        //     loop += 1;
        // }
        //
        /* Finishing touch */
        printf("\n>> Dataset:\n");
        for(int index = 0; index < Size; index++) {
            printf("    [ %.4f |", DataSet[index].Output);
            for(int feature = 1; feature < Features; feature++) {
                printf(" %.4f", DataSet[index].Input[feature]);
            }
            printf(" ]\n");
        }
        // printf("\n>> Linear regression calculating with gradient descent, learning rate %.4f, accepted error %.4f.\n", LearningRate, AcceptedError);
        // printf("   Bias and parameters after %d loops:\n", loop);
        // printf("    [");
        // printf(" %.4f |", Parameters[0]);
        // for(int index = 1; index < Features + 1; index++) {
        //     printf(" %.4f", Parameters[index]);
        // }
        // printf(" ]\n");
    }

    /* Slaves */
    else {
        /* Initializing data  */
        int Features = atoi(argv[1]) + 1;
        int TaskCount = 0;

        /* Receiving task index */
        MPI_Recv(&TaskCount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, NULL);
        if(TaskCount == 0) {
            MPI_Finalize();
            return 0;
        }

        /* Receiving data from master */
        Data* DataSet = calloc(TaskCount, sizeof(Data));
        for(int task = 0; task < TaskCount; task++) {
            Data DataPoint;
            double* Input = (double*)calloc(Features, sizeof(double));
            double Output = 0;
            MPI_Recv(Input, Features, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, NULL);
            MPI_Recv(&Output, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, NULL);
            DataPoint.Input = Input; DataPoint.Output = Output;
        }
        for(int task =0; task < TaskCount; task++) {
            printf("[%d]: %f -", ProcessID, DataSet[task].Output);
            for(int feature = 0; feature < Features; feature++) {
                printf(" %f", DataSet[task].Input[feature]);
            }
            printf("\n");
        }

        // /* Gradient descent */
        // while(1) {
        //     /* Checking for loop exit */
        //     int exit;
        //     MPI_Recv(&exit, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, NULL);
        //     if(exit == 1) { break; }
        //
        //     /* Receiving parameters from master */
        //     double* Parameters = calloc(Features, sizeof(double));
        //     MPI_Recv(Parameters, Features, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, NULL);
        //
        //     /* Doing the calculation for the assigned tasks */
        //     double* Error = calloc(Features, sizeof(double));
        //     for(int task = 0; task < TaskCount; task++) {
        //         for(int feature = 0; feature < Features; feature++) {
        //             Error[task] += DataSet[task].Input[feature] * Parameters[feature];
        //         }
        //         Error[task] -= DataSet[task].Output;
        //     }
        //     double* PartialDerivatives = calloc(Features, sizeof(double));
        //     for(int feature = 0; feature < Features; feature++) { for(int task = 0; task < TaskCount; task++) { PartialDerivatives[feature] += DataSet[task].Input[feature] * Error[task]; } }
        //
        //     /* Sending partial derivatives to master */
        //     MPI_Send(PartialDerivatives, Features, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        // }
    }

    /* Shutting down parallel environment */
    MPI_Finalize();

    /* Procedure exit */
    return 0;
}
/*  */