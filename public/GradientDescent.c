#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

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
  Data* DataSet = NULL;
  *Size = 0;

  /* Processing */
  FILE* File = fopen(FilePath, "r");
  while(!feof(File)) {
    Data DataPoint;
    char* buffer = (char*)calloc(1024, sizeof(char));
    fgets(buffer, 1024, File);
    char* value = strtok(buffer, ",");
    if(value == NULL) {
      break;
    }
    DataSet = (Data*)realloc(DataSet, (*Size + 1) * sizeof(Data));
    DataPoint.Output = atof(value);
    DataPoint.Input = (double*)calloc(Features, sizeof(double));
    DataPoint.Input[0] = 1;
    for(int feature = 1; feature < Features; feature++) {
      value = strtok(NULL, ",");
      if(value == NULL) { break; }
      DataPoint.Input[feature] = atof(value);
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
    int exit = 0;
    FILE* File = fopen(argv[2], "r");
    if(argc != 5) {
      printf("Invalid number of arguments.");
      exit = 1;
    }
    else {
      if(File == NULL) {
        printf("Cannot open file.");
        exit = 1;
      }
      if(atoi(argv[1]) <= 0) {
        printf("Insufficient number of features.");
        exit = 1;
      }
      else if(atof(argv[3]) <= 0) {
        printf("Insufficient learning rate.");
        exit = 1;
      }
      else if(atof(argv[4]) <= 0) {
        printf("Insufficient accepted error value.");
        exit = 1;
      }
    }
    for(int process = 1; process < Processes; process++) {
      MPI_Send(&exit, 1, MPI_INT, process, 0, MPI_COMM_WORLD);
    }
    if(exit == 1) {
      MPI_Finalize();
      return 0;
    }

    /* Initializing time counter */
    double TotalTime = 0;
    double Start, End;
    double TotalTimeWithComp = 0;
    double ProgramStart = MPI_Wtime();

    /* Initializing data  */
    Start = MPI_Wtime();  
    Data* DataSet = NULL;
    int Size = 0;
    int Features = atoi(argv[1]) + 1;
    DataSet = (Data*)parseFile(&Size, Features, argv[2]);
    End = MPI_Wtime();  
    TotalTime += End - Start;  

    /* Assigning tasks */
    Start = MPI_Wtime();  
    TaskAssignment* Tasks = (TaskAssignment*)calloc(Processes, sizeof(TaskAssignment));
    int quotient = Size / Processes; int remainder = Size % Processes;
    End = MPI_Wtime();  
    TotalTime += End - Start;  
    for(int process = 0; process < Processes; process++) {
      Start = MPI_Wtime();  
      Tasks[process].Start = quotient * process + (process < remainder ? process : remainder);
      Tasks[process].End = Tasks[process].Start + (process < remainder ? quotient : quotient - 1);
      int TaskCount = Tasks[process].End - Tasks[process].Start + 1;
      End = MPI_Wtime();  
      TotalTime += End - Start;  
      MPI_Send(&TaskCount, 1, MPI_INT, process, 0, MPI_COMM_WORLD);
    }
    Start = MPI_Wtime();  
    int TaskCount = Tasks[0].End - Tasks[0].Start + 1;
    if(Size < Processes) {
      Processes = Size;
    }
    End = MPI_Wtime();  
    TotalTime += End - Start;  

    /* Sending data to slaves */
    for(int process = 1; process < Processes; process++) {
      for(int task = Tasks[process].Start; task <= Tasks[process].End; task++) {
        MPI_Send(DataSet[task].Input, Features, MPI_DOUBLE, process, 0, MPI_COMM_WORLD);
        MPI_Send(&DataSet[task].Output, 1, MPI_DOUBLE, process, 0, MPI_COMM_WORLD);
      }
    }

    /* Gradient Descent */
    Start = MPI_Wtime();  
    double LearningRate = atof(argv[3]);
    double AcceptedError = atof(argv[4]);
    double* Parameters = (double*)calloc(Features, sizeof(double));
    double* Derivatives = (double*)calloc(Features, sizeof(double));
    for(int feature = 0; feature < Features; feature++) {
      Parameters[feature] = 1;
      Derivatives[feature] = 1;
    }
    int loop = 0;
    End = MPI_Wtime();  
    TotalTime += End - Start;  
    while(1) {
      /* Checking for loop exit */
      Start = MPI_Wtime();  
      int exit = 1;
      for(int feature = 0; feature < Features; feature++) {
        if(fabs(Derivatives[feature]) > AcceptedError) { exit = 0; }
      }
      exit = exit == 1 || loop == 100000000 ? 1 : 0;
      End = MPI_Wtime();  
      TotalTime += End - Start;  
      for(int process = 1; process < Processes; process++) {
        MPI_Send(&exit, 1, MPI_INT, process, 0, MPI_COMM_WORLD);
      }
      Start = MPI_Wtime();  
      if(exit == 1) {
        break;
      }
      End = MPI_Wtime();  
      TotalTime += End - Start;  

      /* Sending current parameters to slave */
      for(int process = 1; process < Processes; process++) {
        MPI_Send(Parameters, Features, MPI_DOUBLE, process, 0, MPI_COMM_WORLD);
      }

      /* Doing the calculation for the assigned tasks */
      Start = MPI_Wtime();  
      double* Error = (double*)calloc(Features, sizeof(double));
      for(int task = 0; task < TaskCount; task++) {
        for(int feature = 0; feature < Features; feature++) {
          Error[task] += DataSet[task].Input[feature] * Parameters[feature];
        }
        Error[task] -= DataSet[task].Output;
      }
      for(int feature = 0; feature < Features; feature++) {
        Derivatives[feature] = 0;
        for(int task = 0; task < TaskCount; task++) {
          Derivatives[feature] += DataSet[task].Input[feature] * Error[task];
        }
      }
      End = MPI_Wtime();  
      TotalTime += End - Start;  

      /* Receiving partial derivatives back from slaves and calculating derivatives */
      for(int process = 1; process < Processes; process++) {
        Start = MPI_Wtime();  
        double* PartialDerivatives = (double*)calloc(Features, sizeof(double));
        End = MPI_Wtime();  
        TotalTime += End - Start;  
        MPI_Recv(PartialDerivatives, Features, MPI_DOUBLE, process, 0, MPI_COMM_WORLD, NULL);
        Start = MPI_Wtime();  
        for(int feature = 0; feature < Features; feature++) {
          Derivatives[feature] += PartialDerivatives[feature];
        }
        End = MPI_Wtime();  
        TotalTime += End - Start;  
      }

      /* Updating parameters */
      Start = MPI_Wtime();  
      for(int feature = 0; feature < Features; feature++) {
        Parameters[feature] -= LearningRate * Derivatives[feature];
      }
      End = MPI_Wtime();  
      TotalTime += End - Start;  

      /* Updating loop count */
      Start = MPI_Wtime();  
      loop += 1;
      End = MPI_Wtime();
      TotalTime += End - Start;
    }

    /* Measuring time */
    double ProgramEnd = MPI_Wtime();
    TotalTimeWithComp = ProgramEnd - ProgramStart;

    /* Finishing touch */
    printf("\n>> Linear regression calculating with gradient descent, learning rate %f, accepted error %f.\n", LearningRate, AcceptedError);
    printf("   Bias and parameters after %d loops:\n", loop);
    printf("  [");
    printf(" %.4f |", Parameters[0]);
    for(int index = 1; index < Features; index++) {
      printf(" %.4f", Parameters[index]);
    }
    printf(" ]\n");
    printf("\n>> Total time: %f(s), without comp: %f(s).\n", TotalTimeWithComp, TotalTime);
    printf("\n");
  }

  /* Slaves */
  else {
    /* Validating terminal arguments */
    int exit = 0;
    MPI_Recv(&exit, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, NULL);
    if(exit == 1) {
      MPI_Finalize();
      return 0;
    }

    /* Initializing data */
    int Features = atoi(argv[1]) + 1;
    int TaskCount = 0;

    /* Receiving task index */
    MPI_Recv(&TaskCount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, NULL);
    if(TaskCount == 0) {
      MPI_Finalize();
      return 0;
    }

    /* Receiving data from master */
    Data* DataSet = (Data*)calloc(TaskCount, sizeof(Data));
    for(int task = 0; task < TaskCount; task++) {
      Data DataPoint;
      double* Input = (double*)calloc(Features, sizeof(double));
      double Output = 0;
      MPI_Recv(Input, Features, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, NULL);
      MPI_Recv(&Output, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, NULL);
      DataPoint.Input = Input; DataPoint.Output = Output;
      DataSet[task] = DataPoint;
    }

    /* Gradient descent */
    while(1) {
      /* Checking for loop exit */
      int exit;
      MPI_Recv(&exit, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, NULL);
      if(exit == 1) {
        break;
      }

      /* Receiving parameters from master */
      double* Parameters = (double*)calloc(Features, sizeof(double));
      MPI_Recv(Parameters, Features, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, NULL);

      /* Doing the calculation for the assigned tasks */
      double* Error = (double*)calloc(Features, sizeof(double));
      for(int task = 0; task < TaskCount; task++) {
        for(int feature = 0; feature < Features; feature++) {
          Error[task] += DataSet[task].Input[feature] * Parameters[feature];
        }
        Error[task] -= DataSet[task].Output;
      }
      double* PartialDerivatives = (double*)calloc(Features, sizeof(double));
      for(int feature = 0; feature < Features; feature++) {
        PartialDerivatives[feature] = 0;
        for(int task = 0; task < TaskCount; task++) {
          PartialDerivatives[feature] += DataSet[task].Input[feature] * Error[task];
        }
      }

      /* Sending partial derivatives to master */
      MPI_Send(PartialDerivatives, Features, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
  }

  /* Shutting down parallel environment */
  MPI_Finalize();

  /* Procedure exit */
  return 0;
}
/*  */