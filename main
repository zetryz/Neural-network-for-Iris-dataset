//Testing multiple rows


#include <iostream>
#include <windows.h>
#include <fstream>
#include <math.h>
#include <string.h>
#include <sstream>
#include <vector>
#include <stdio.h> 
#include <conio.h>
using namespace std;

double rms(double x[], int n);

class neuron {
	//input neurons 0 
	//hidden neurons 1
	//output neurons 2
	static const int maxLimit = 800;
	double value;		//for input and hidden layers
//	double input;		//for input layer
	double sum;			// for hidden and output layer
	double delta;
	double weights[maxLimit];
	double prev_iter_delta_weight[maxLimit];
	static double eta;
	static double momentum;
	public:
		neuron() {
			value = 0;
			sum = 0;
			for (int i = 0;i < maxLimit;i++) {
				weights[i] = 0;
				prev_iter_delta_weight[i] = 0;
			}
		}
		neuron(double x) {
			value = x;
			sum = 0;
			for (int i = 0;i < maxLimit;i++) {
				weights[i] = 0;
				prev_iter_delta_weight[i] = 0;
			}
		}
/*		~neuron() {
			delete weights;
		}*/
		double get_value() {
			return value;
		}
		void set_value(double x) {		//used for input layer
			value = x;
		}
		double get_weights(int x) {
			return weights[x];
		}
		//calculate random weights
		void calculate_random_weights(int prevLayerDimension) {
			for (int i = 0;i < prevLayerDimension;i++) {
				weights[i] = rand() / double(RAND_MAX);  
			}
		}
		double calculate_sum(neuron* prevLayer, int prevLayerDimension, int add_bias) {
				sum = 0;
				for (int i = 0;i < prevLayerDimension + add_bias;i++) {
				//	cout << i << " => " << weights[i] << endl;
					sum = sum + prevLayer[i].get_value() * weights[i];			
				}	
				return sum;
		}
		double calculate_output() {
			/*Sigmoid*/
		//	value = 1/(1 + exp(-0.1 * sum));
			
			/*Tanh*/
			value = tanh(sum);
		}
		double calculate_output_delta(double& desired_output) {
			delta = desired_output - value; 
			//return output_delta;
		}
		double calculate_delta(neuron* nextLayer, int nextLayerDimension, int neuronNr) {
			delta = 0;
			for (int i = 0;i < nextLayerDimension;++i) {
				delta = delta + (nextLayer[i].get_delta() * nextLayer[i].get_weights(neuronNr));
			}
			return delta;
		}
		double get_delta() {
			return delta;
		}
		void set_delta(double x) {
			delta = x;
		}
		void recalculate_weights(neuron* prevLayer, int prevLayerDimension) {
			for (int i = 0;i < prevLayerDimension;i++) {
				//double last_weight_value = weights[i];
				
				/*Sigmoid*/
			//	double delta_weight = eta * delta * (value * (1 - value)) * prevLayer[i].get_value();
				/*Tanh*/
				double delta_weight = eta * delta * (1 - pow(value, 2)) * prevLayer[i].get_value();
				
				weights[i] = weights[i] + delta_weight + prev_iter_delta_weight[i] * momentum;
				prev_iter_delta_weight[i] = delta_weight;
			//	cout << weights[i] << endl;
			}
		}
		
};
double neuron::eta = 0.01;
double neuron::momentum = 0.7;

int main() {
	//what to print
	bool print_to_outfile = 0;
	bool print_weights_to_outfile = 0;
	
	////global variables
	int i, j, k, l, m, n, o, r, u;
	ofstream outFile, outFileMB;
    outFile.open ("resultsOut.csv");
    outFileMB.open("resultsOut2.csv");
    
    ////network topology
	int number_of_layers = 5;
	int max_neurons_per_layer = 800;
	int input_size = 4;			//number of objects
	int output_size = 3;		//number of categories/groups/classes
//	int hidden_layer_size = 3;
	int* neurons_per_layer = new int[number_of_layers];
	//int neurons_per_layer[number_of_layers] = {input_size, hidden_layer_size, output_size};
	neurons_per_layer[0] = input_size;
	neurons_per_layer[1] = 12;
	neurons_per_layer[2] = 20;
	neurons_per_layer[3] = 12;
	neurons_per_layer[4] = output_size;
	
	///learning & testing parameters
	int number_of_rows = 150;		//number of learning examples
	int test_number_of_rows = 150;	//number of testing samples
	int nr_of_iter = 1000;	// nr. of iterations X all learning set
	int nr_of_updates = 1;
	int add_bias = 1;
	
//**************	
	//	neuron network[number_of_layers][max_neurons_per_layer];
	neuron** network = new neuron*[number_of_layers];
	for (i = 0;i < number_of_layers;i++) {
		network[i] = new neuron[max_neurons_per_layer];	
	}
	//assign values to allocated memory
	for (i = 0;i < number_of_layers;i++) {
		for (j = 0;j < max_neurons_per_layer;j++) {
			network[i][j] = 0;
		}
	}
//**************		
	////Get input from CSV file
		
	//	double input[number_of_rows][input_size];
	double** input = new double*[number_of_rows];
	for (i = 0;i < number_of_rows;i++) {
		input[i] = new double[input_size];		
	}
	
	ifstream inputFile;
	string lines, word;
	//	vector<string> rowLabels;
	double input_values;
	inputFile.open("input.csv");
	for (i = 0;i < number_of_rows;i++) {
		getline(inputFile, lines);
		stringstream s(lines);
		k = 0;
		while (getline(s, word, ',')) {
//			rowLabels.push_back(word);
			input_values = atof(word.c_str());
			//cout << input_values << "  ";
			input[i][k] = input_values;
			k++;
		}
		cout << endl;
	}
	inputFile.close();
	
//	for (int i = 0;i < input_size;i++) {
//		network[0][i].set_value(input[0][i]);						//input
	//	cout << input[0][i] << " ";
//	}
	
	////get output from CSV file
	
//	double output4 = 1;		//Yes
//	double output5 = 0;		//No
//	double output[2] = {1, 0}; 
	
//	double output[number_of_rows][output_size];
	double** output = new double*[number_of_rows];
	for (i = 0;i < number_of_rows;i++) {
		output[i] = new double[output_size];	
	}
	
	ifstream outputFile;
	outputFile.open("output.csv");
	for (i = 0;i < number_of_rows;i++) {
		getline(outputFile, lines);
		stringstream s(lines);
		k = 0;
		while (getline(s, word, ',')) {
//			rowLabels.push_back(word);
			input_values = atof(word.c_str());
			//cout << input_values << "  ";
			output[i][k] = input_values;
			k++;
		}
		cout << endl;
	}
	outputFile.close();
//**************		
	for (int i = 0;i < output_size;i++) {
		network[number_of_layers - 1][i].set_value(output[0][i]);						//output
	//	cout << input[0][i] << " ";
	}
	
//**************		
	////Write CSV file  -> excel 
    if (print_to_outfile == 1) outFile << "Iter,";
    outFileMB << "Iter,";
    		//neurons -> excel 
    if (print_to_outfile == 1) {
	    for (j = 0;j < number_of_layers;j++) {				//pentru fiecare strat
	    	for (i = 0;i < neurons_per_layer[j];i++) {			//ptr fiecare neuron
	    		outFile << "Neuron" << j << i << ",";
			}
			if ((add_bias == 1)&&(j < number_of_layers - 1)) 					//add 1 more layer Title for bias neurons
				outFile << "Neuron" << j << i << ",";
		}
	}
	
			//output layer deltas -> excel 
	if (print_to_outfile == 1) {
	for (j = 0;j < neurons_per_layer[number_of_layers - 1];j++) {
		outFile << "Delta" << number_of_layers - 1 << j << ",";
	}
	}
//	for (j = 0;j < neurons_per_layer[number_of_layers - 1];j++) {
///		outFile << "RMSD" << number_of_layers - 1 << j << ",";
///		outFileMB << "RMSD" << number_of_layers - 1 << j << ",";
//	}
	
			//hidden layers deltas -> excel
	if (print_to_outfile == 1) {
		for (j = number_of_layers - 2;j > 0;j--) { 
			for (k = 0;k < neurons_per_layer[j];k++) {
				outFile << "Delta" << j << k << ",";
			}
		}		
	}
	
			//weights -> excel
	if (print_weights_to_outfile == 1) {
		for (l = 1;l < number_of_layers;l++) {
				for (j = 0;j < neurons_per_layer[l];j++) {
					for (i = 0;i < neurons_per_layer[l-1] + add_bias; i++){	
						outFile << "weight" << l << j << i << ",";															
					}
				}
		} 
	}
	
	if (print_to_outfile == 1) outFile << "\n";
	outFileMB << "\n";
	
//**************	
	////Calculate random weights

	for (i = 1;i < number_of_layers;i++) {
		for (k = 0;k < neurons_per_layer[i] + add_bias;k++) {
			network[i][k].calculate_random_weights(neurons_per_layer[i - 1]);  
			for (j = 0;j < neurons_per_layer[i-1];j++) {
				//cout << "weight"<< i << k << j << " " << network[i][k].get_weights(j) << endl;	
			}
		}
	}
	
	
//**************
	////Training
	o = 0;
	double global_error = 0;
	double normalize = 0;
	for (i = 0;i < nr_of_iter;i++) {
		outFileMB << i << ",";
		global_error = 0;
		cout << "************* Iter "<< i << " *************" << endl << endl;
		
		//Iterate for each pattern m number of times m = 10, update weights 10 times before going to next row
		
		for (m = 0;m < number_of_rows;m++) {
		//	cout << "************* Row "<< m << " *************" << endl << endl;
			for (n = 0;n < nr_of_updates;n++) {
			//	cout << "************* Update "<< n << " *************" << endl << endl;
				
				//Set input values for each row
				for (k = 0;k < input_size;k++) {
					network[0][k].set_value(input[m][k]);						
				}
				if (add_bias == 1) network[0][k].set_value(1);
				
				//Set biases for hidden layers
				if (add_bias == 1) {
					for (l = 1;l < number_of_layers - 1;l++) {
						network[l][neurons_per_layer[l]].set_value(1);
					}
				}
				//Write input values to CSV
				if (print_to_outfile == 1) outFile << o << ",";
				o++;
				if (print_to_outfile == 1) {
					for (k = 0;k < neurons_per_layer[0];k++) { 
						outFile  << network[0][k].get_value() << ",";
					}
					if (add_bias == 1) outFile  << network[0][k].get_value() << ",";
				}
				//calculate outputs	
				for (l = 1;l < number_of_layers;l++) {
					for (k = 0;k < neurons_per_layer[l];k++) {
						//cout << "sum" << l << k << " = " <<  << endl
						network[l][k].calculate_sum(network[l-1], neurons_per_layer[l - 1], add_bias);	
						network[l][k].calculate_output();
						//cout << "neuron" << l << k << " = " << network[l][k].get_value() << endl;
						if (print_to_outfile == 1) outFile << network[l][k].get_value() << ",";	//										
					}
					if ((add_bias == 1)&&(l < number_of_layers - 1)&&(print_to_outfile == 1)) outFile << network[l][k].get_value() << ",";
				}
				
				//calculate deltas for output layer
				for (l = 0;l < neurons_per_layer[number_of_layers - 1];l++){
					network[number_of_layers - 1][l].calculate_output_delta(output[m][l]);
				//	cout << output[m][l] ;
					//cout << "delta" << l << " = " << network[number_of_layers - 1][l].get_delta() << endl;
					if (print_to_outfile == 1) outFile << network[number_of_layers - 1][l].get_delta() << ",";		
				//	sum_of_deltas_for_all_rows[l] += network[number_of_layers - 1][l].get_delta();
				//	sum_of_deltas_for_all_rows[l] += network[number_of_layers - 1][l].get_delta();	
					global_error += pow(network[number_of_layers - 1][l].get_delta(), 2);	
				}
						
				//calculate  hidden layers deltas
				for (l = number_of_layers - 2;l > 0;l--) { 
					for (k = 0;k < neurons_per_layer[l];k++) {
						//cout << "delta" << l << k << " = "<<  << endl
						network[l][k].calculate_delta(network[l+1], neurons_per_layer[l+1], k);
						if (print_to_outfile == 1) outFile << network[l][k].get_delta() << ",";	
					}
				}
				
				//recalculate weights
				for (l = 1;l < number_of_layers;l++) {
					for (j = 0;j < neurons_per_layer[l];j++) {
						if (print_weights_to_outfile == 1) {
							for (u = 0;u < neurons_per_layer[l - 1] + add_bias; u++){	
								outFile << network[l][j].get_weights(u) << ","; 
							//	cout << "weight" << l << j << i << "= " << network[l][j].get_weights(i) << endl;																//+1 Bias
							}
						}
						network[l][j].recalculate_weights(network[l - 1], neurons_per_layer[l - 1] + add_bias);
						
					}
				}
				outFile << "\n";
			}
		}
		global_error = global_error/number_of_rows;
		outFileMB << global_error << ",";
		cout << endl << endl;
		
		outFileMB << "\n";
	} 
	outFile.close();
	outFileMB.close();
	//Save Model
	
	//Test Model 
//	cout << "Press 1 for testing ..."<< endl;
//	char c;
//	getch();
	// Open input test data file and output test data file 
	//Input Test data
	
	ofstream testDataResults;
	testDataResults.open("TestdataResults.csv");
	//Data set to test
	ifstream inputTestFile;
    inputTestFile.open("input.csv");
    
 //  double test_input[test_number_of_rows][input_size];
	double** test_input = new double*[test_number_of_rows];
	for (i = 0;i < test_number_of_rows;i++) {
		test_input[i] = new double[input_size];	
	}
	for (i = 0;i < test_number_of_rows;i++) {
		getline(inputTestFile, lines);
		stringstream s(lines);
		k = 0;
		while (getline(s, word, ',')) {
//			rowLabels.push_back(word);
			input_values = atof(word.c_str());
		//	cout << input_values << "  ";
			test_input[i][k] = input_values;
			k++;
		}
		cout << endl;
	}
	inputTestFile.close();

	//Run input through the trained model
	for (i = 0;i < test_number_of_rows;i++) {
		//network.TakeInput();
		for (k = 0;k < input_size;k++) {
			network[0][k].set_value(test_input[i][k]);	
			//cout << network[0][k].get_value() << " ";					
		}
	//	cout << endl << endl;
		//network.feedforward();
		//calculate outputs	
		for (l = 1;l < number_of_layers;l++) {
			for (k = 0;k < neurons_per_layer[l];k++) {
				//cout << "sum" << l << k << " = " << << endl;
				network[l][k].calculate_sum(network[l-1], neurons_per_layer[l - 1], add_bias); 	
				network[l][k].calculate_output();
			//	cout << "neuron" << l << k << " = " << network[l][k].get_value() << endl;
				if (l == (number_of_layers - 1)) testDataResults << network[l][k].get_value() << ",";	//										
			}
		}
		testDataResults << "\n";
	//network.GetResults();
	}
    //Compare and print results

    testDataResults.close();
/*    
    double input[2] = {0, 1};
	double target_data[2] = {0, 1} 
	*/
	// 1 0 0 GREEN
	// 0 1 0 RED
	// 0 0 1 PURPLE
	
	inputTestFile.close();

	//Delete dynamically allocated memory	
	//input
	for (i = 0;i < number_of_rows;i++) {
		delete[] input[i];
	}
	delete[] input;
	//output
	for (i = 0;i < number_of_rows;i++) {
		delete[] output[i];
	}
	delete[] output;
	//test_input
	for (i = 0;i < test_number_of_rows;i++) {
		delete[] test_input[i];
	}
	delete[] test_input;
	//network
	for (i = 0;i < number_of_layers;i++) {
		delete[] network[i];
	}
	delete[] network;
	
	return 0;
}

double rms(double* x, int n)
{
	double sum = 0;

	for (int i = 0; i < n; i++)
		sum += pow(x[i], 2);

	return sqrt(sum / n);
}
