#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Data declaration */
typedef struct {
    double *input;
    double output;
} Data;
/*  */

/* Entry procedure */
int main(int argc, char** argv) {
    // Initializing time counter
    double TotalTime = 0;
    clock_t ProgramStart = clock();

    // Initializing
    Data** DataSet = (Data**)calloc(1024, sizeof(Data*));
    int Size = 0;
    int Features = atoi(argv[1]);

    // Parse input from file
    FILE* File = fopen(argv[2], "r");
    while(!feof(File)) {
        Data* DataPoint = (Data*)calloc(1, sizeof(Data));
        char* buffer = (char*)calloc(1024, sizeof(char));
        fgets(buffer, 1024, File);
        char* value = strtok(buffer, ",");
        if(value == NULL) break;
        DataPoint->output = atof(value);
        DataPoint->input = (double*)calloc(Features + 1, sizeof(double));
        DataPoint->input[0] = 1;
        for(int index = 1; index < Features + 1; index++) {
            value = strtok(NULL, ",");
            if(value == NULL) break;
            DataPoint->input[index] = atof(value);
        }
        DataSet[Size] = DataPoint;
        Size += 1;
    }
    fclose(File);

    // Gradient descent
    double* Parameters = (double*)calloc(Features + 1, sizeof(double));
    for(int index = 0; index < Features + 1; index++) {
        Parameters[index] = 1;
    }
    double* Derivatives = (double*)calloc(Features + 1, sizeof(double));
    for(int index = 0; index < Features + 1; index++) {
        Derivatives[index] = 1;
    }
    double* PredictedOutput = (double*)calloc(Size, sizeof(double));
    double LearningRate = 0.000001;
    double AcceptedError = 0.001;
    int LoopCount = 0;
    while(1) {
        // Checking for loop exit
        bool exit = true;
        for(int index = 0; index < Features + 1; index++) {
            if(fabs(Derivatives[index]) > AcceptedError) {
                exit = false;
            }
        }
        if(exit == true || LoopCount == 10000000) break;
        // Calculating predicted output
        for(int datapoint_index = 0; datapoint_index < Size; datapoint_index++) {
            PredictedOutput[datapoint_index] = 0;
            for(int parameter_index = 0; parameter_index < Features + 1; parameter_index++) {
                double input = DataSet[datapoint_index]->input[parameter_index];
                double parameter = Parameters[parameter_index];
                PredictedOutput[datapoint_index] += input * parameter;
            }
        }
        // Calculating derivatives
        for(int parameter_index = 0; parameter_index < Features + 1; parameter_index++) {
            Derivatives[parameter_index] = 0;
            for(int datapoint_index = 0; datapoint_index < Size; datapoint_index++) {
                double input = DataSet[datapoint_index]->input[parameter_index];
                double expected_output = DataSet[datapoint_index]->output;
                double predicted_output = PredictedOutput[datapoint_index];
                Derivatives[parameter_index] += input * (predicted_output - expected_output);
            }
        }
        // Updating parameters
        for(int index = 0; index < Features + 1; index++) {
            Parameters[index] -= LearningRate * Derivatives[index];
        }
        LoopCount += 1;
    }

    // Measuring time
    clock_t ProgramEnd = clock();
    TotalTime = (double)(ProgramEnd - ProgramStart) / CLOCKS_PER_SEC;

    // Finishing touch
    printf("\n>> Linear regression calculating with gradient descent, learning rate %.4f, accepted error %.4f.\n", LearningRate, AcceptedError);
    printf("   Bias and parameters after %d loops in %f(s):\n", LoopCount, TotalTime);
    printf("    [");
    printf(" %.4f |", Parameters[0]);
    for(int index = 1; index < Features + 1; index++) {
        printf(" %.4f", Parameters[index]);
    }
    printf(" ]\n");

    // Procedure exit
    return 0;
}
/*  */