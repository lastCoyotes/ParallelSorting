/*******************************************************************************/
/* Sorting Timing Program -- C++ Serial/OpenMP Implementation                  */
/* 2021 -- Tristan Boler -- tab0037@uah.edu                                    */
/* Much inspiration taken from various source files by Dr. B. Earl Wells       */
/*******************************************************************************/
/*
  Compilation on dmc.asc.edu (and jetson cluster)
   GNU Compiler
   module load gcc
   g++ sorting_speed.cpp SortTimeResults.hpp -o sorting_speed_gnu -std=c++20 -fopenmp
To execute:
   
   GNU compiler
   use run_script sorting_speed_gnu.sh
*/
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <random>
#include <limits>
#include <algorithm>
#include <string>
#include <concepts>
#include <iterator>
#include <chrono>
#include <vector>
#include <cmath>
#include <functional>
#include <memory>
//#include <execution>
#include <omp.h> // allows use of OpenMP API Calls //

#include "SortTimeResults.hpp"

template<typename T>
requires std::sortable<T>
void insertionSort(const T &begin, const T &end, const bool dir = true) {
	for (auto i(begin); i != end; ++i) {
		if (dir) {
			std::rotate(std::upper_bound(begin, i, *i), i, i+1);
		} else {
			std::rotate(std::upper_bound(begin, i, *i, std::greater<>()), i, i+1);
		}
	}
}

template<typename T>
requires std::sortable<T>
void mergeSort(const T &begin, const T &end, const int &smallSortLimit, const SortMethod method = SortMethod::Insertion) {
	const auto distance = std::distance(begin, end);
	if (distance <= 1) {
		return;
	}
	if (distance < smallSortLimit) {
		if (method == SortMethod::Insertion) {
			insertionSort(begin, end);
		} else if (method == SortMethod::stdSort) {
			std::sort(begin, end);
		}
		return;
	}
	const T middleIt = std::next(begin, distance / 2);
	#pragma omp task
	mergeSort(begin, middleIt, smallSortLimit, method);
	#pragma omp task
	mergeSort(middleIt, end, smallSortLimit, method);
	#pragma omp taskwait
	std::inplace_merge(begin, middleIt, end);
}

template<typename T>
requires std::sortable<T>
void serialMergeSort(const T &begin, const T &end, const int &smallSortLimit, const SortMethod method = SortMethod::Insertion) {
	const auto distance = std::distance(begin, end);
	if (distance <= 1) {
		return;
	}
	if (distance < smallSortLimit) {
		if (method == SortMethod::Insertion) {
			insertionSort(begin, end);
		} else if (method == SortMethod::stdSort) {
			std::sort(begin, end);
		}
		return;
	}
	const T middleIt = std::next(begin, distance / 2);
	serialMergeSort(begin, middleIt, smallSortLimit, method);
	serialMergeSort(middleIt, end, smallSortLimit, method);
	std::inplace_merge(begin, middleIt, end);
}

//https://stackoverflow.com/questions/1505675/power-of-an-integer-in-c
constexpr int myPow(int x, unsigned int p)
{
  if (p == 0) return 1;
  if (p == 1) return x;
  
  int tmp = myPow(x, p/2);
  if (p%2 == 0) return tmp * tmp;
  else return x * tmp * tmp;
}

int greatestPowerOfTwoLessThan(int n) {
	int k = 1;
	while (k > 0 && k < n) {
		k <<= 1;
	}
	return k >> 1;
}

template<typename T>
requires std::sortable<T>
void bitonicMerge(const T &begin, const T &end, const bool dir) {
	const int n = std::distance(begin, end);
	if (n > 1) {
		const int m = greatestPowerOfTwoLessThan(n);
		for (T i(begin); i < std::prev(end, m); i++) {
			const T j = std::next(i, m);
			if (dir == (*i > *j)) {
				std::iter_swap(i, j);
			}
		}
		const T middle = std::next(begin, m);
		bitonicMerge(begin, middle, dir);
		bitonicMerge(middle, end, dir);
	}
}

template<typename T>
requires std::sortable<T>
//dir ascending = true
void bitonicSort(const T &begin, const T &end, const bool dir, const int smallSortLimit,
		const SortMethod method = SortMethod::Insertion) {
	const int n = std::distance(begin, end);
	if (n < smallSortLimit) {
		if (method == SortMethod::Insertion) {
			insertionSort(begin, end, dir);
		} else if (method == SortMethod::stdSort) {
			if (dir) {
				std::sort(begin, end);
			} else {
				std::sort(begin, end, std::greater<>());
			}
		}
	} else {
		const T middle = std::next(begin, n/2);
		#pragma omp task
		bitonicSort(begin, middle, !dir, smallSortLimit, method);
		#pragma omp task
		bitonicSort(middle, end, dir, smallSortLimit, method);
		#pragma omp taskwait
		bitonicMerge(begin, end, dir);
	}
}

template<typename T>
requires std::sortable<T>
//dir ascending = true
void serialBitonicSort(const T &begin, const T &end, const bool dir, const int smallSortLimit,
		const SortMethod method = SortMethod::Insertion) {
	const int n = std::distance(begin, end);
	if (n < smallSortLimit) {
		if (method == SortMethod::Insertion) {
			insertionSort(begin, end, dir);
		} else if (method == SortMethod::stdSort) {
			if (dir) {
				std::sort(begin, end);
			} else {
				std::sort(begin, end, std::greater<>());
			}
		}
	} else {
		const T middle = std::next(begin, n/2);
		serialBitonicSort(begin, middle, !dir, smallSortLimit, method);
		serialBitonicSort(middle, end, dir, smallSortLimit, method);
		bitonicMerge(begin, end, dir);
	}
}

//https://stackoverflow.com/questions/14539867/how-to-display-a-progress-indicator-in-pure-c-c-cout-printf
void printProgress(float progress) {
	const int barWidth = 70;

	std::cout << "[";
	const int pos = barWidth * progress;
	for (int i = 0; i < barWidth; ++i) {
		if (i < pos) std::cout << "=";
		else if (i == pos) std::cout << ">";
		else std::cout << " ";
	}
	std::cout << "] " << int(progress * 100.0) << " %\r";
	std::cout.flush();
}

template<typename T>
void testSerialBitonicInsertionSort(const std::vector<T> &originalVector, 
		const std::vector<long> &sortLengths, 
		const std::vector<int> &sortThresholds, 
		std::vector<std::unique_ptr<SortTimeResult>> &results,
		const int threads,
		const int run)
{
	const int maxTestCasesPerSort = sortLengths.size() * sortThresholds.size();
	//int testCase = 0;
	//std::cout << "testing serial bitonic with insertion sort" << std::endl;
	#pragma omp parallel for collapse(2) schedule(dynamic,1)
	for (const long sortLength : sortLengths) {
		for (const int sortThreshold : sortThresholds) {
			if (sortThreshold > sortLength) {
				//#pragma omp critical (progress)
				//testCase++;
				continue;
			}
			std::vector<T> sortVector(originalVector.begin(), std::next(originalVector.begin(), sortLength));
		
			auto startTime = std::chrono::steady_clock::now();
			serialBitonicSort(sortVector.begin(), sortVector.end(), true, sortThreshold);
			auto stopTime = std::chrono::steady_clock::now();
			std::chrono::duration<double> diff = stopTime - startTime;
			if (std::is_sorted(sortVector.begin(), sortVector.end())) {
				MixedSortTimeResult result(SortType::Serial, 1, sortLength, sortThreshold, SortMethod::Bitonic,
						SortMethod::Insertion, diff, run);
				#pragma omp critical (emplace)
				results.emplace_back(std::make_unique<MixedSortTimeResult>(result));
			}
			//#pragma omp critical (progress)
			//printProgress((float) ++testCase / maxTestCasesPerSort); 
		}
	}
	//std::cout << std::endl;
}

template<typename T>
void testSerialBitonicStdSort(const std::vector<T> &originalVector, 
		const std::vector<long> &sortLengths, 
		const std::vector<int> &sortThresholds, 
		std::vector<std::unique_ptr<SortTimeResult>> &results,
		const int threads,
		const int run)
{
	const int maxTestCasesPerSort = sortLengths.size() * sortThresholds.size();
	//int testCase = 0;
	//std::cout << "testing serial bitonic with std sort" << std::endl;
	#pragma omp parallel for schedule(dynamic,1) collapse(2)
	for (const long sortLength : sortLengths) {
		for (const int sortThreshold : sortThresholds) {
			if (sortThreshold > sortLength) {
				//#pragma omp critical (progress)
				//testCase++;
				continue;
			}
			std::vector<T> sortVector(originalVector.begin(), std::next(originalVector.begin(), sortLength));
		
			auto startTime = std::chrono::steady_clock::now();
			serialBitonicSort(sortVector.begin(), sortVector.end(), true, sortThreshold, SortMethod::stdSort);
			auto stopTime = std::chrono::steady_clock::now();
			std::chrono::duration<double> diff = stopTime - startTime;
			if (std::is_sorted(sortVector.begin(), sortVector.end())) {
				MixedSortTimeResult result(SortType::Serial, 1, sortLength, sortThreshold, SortMethod::Bitonic,
						SortMethod::stdSort, diff, run);
				#pragma omp critical (emplace)
				results.emplace_back(std::make_unique<MixedSortTimeResult>(result));
			}
			//#pragma omp critical (progress)
			//printProgress((float) ++testCase / maxTestCasesPerSort); 
		}
	}
	//std::cout << std::endl;
}

template<typename T>
void testParallelBitonicInsertionSort(const std::vector<T> &originalVector, 
		const std::vector<long> &sortLengths, 
		const std::vector<int> &sortThresholds, 
		std::vector<std::unique_ptr<SortTimeResult>> &results,
		const int totalThreads,
		const int threads,
		const int run)
{
	const int maxTestCasesPerSort = sortLengths.size() * sortThresholds.size();
	//int testCase = 0;
	//std::cout << "Testing parallel bitonic sort with insertion sort and " << threads << " threads" << std::endl;
	#pragma omp parallel for collapse(2) if(threads<totalThreads) num_threads(totalThreads/threads) schedule(dynamic,1)
	for (const long sortLength : sortLengths) {
		for (const int sortThreshold : sortThresholds) {
			if (sortThreshold > sortLength) {
				//#pragma omp critical (progress)
				//testCase++;
				continue;
			}
			std::vector<T> sortVector(originalVector.begin(), std::next(originalVector.begin(), sortLength));
		
			auto startTime = std::chrono::steady_clock::now();
			#pragma omp parallel num_threads(threads)
			{
				#pragma omp single
				{
					bitonicSort(sortVector.begin(), sortVector.end(), true, sortThreshold);
				}
			}
			auto stopTime = std::chrono::steady_clock::now();
			std::chrono::duration<double> diff = stopTime - startTime;
			if (std::is_sorted(sortVector.begin(), sortVector.end())) {
				MixedSortTimeResult result(SortType::Parallel, threads, sortLength, sortThreshold, SortMethod::Bitonic,
						SortMethod::Insertion, diff, run);
				#pragma omp critical (emplace)
				results.emplace_back(std::make_unique<MixedSortTimeResult>(result));
			}
			
			//#pragma omp critical (progress)
			//printProgress((float) ++testCase / maxTestCasesPerSort);
		}
	}
	//std::cout << std::endl;
}

template<typename T>
void testParallelBitonicStdSort(const std::vector<T> &originalVector, 
		const std::vector<long> &sortLengths, 
		const std::vector<int> &sortThresholds, 
		std::vector<std::unique_ptr<SortTimeResult>> &results,
		const int totalThreads,
		const int threads,
		const int run)
{
	const int maxTestCasesPerSort = sortLengths.size() * sortThresholds.size();
	//int testCase = 0;
	//std::cout << "Testing parallel bitonic sort with std::sort and " << threads << " threads" << std::endl;
	#pragma omp parallel for collapse(2) if(threads<totalThreads) num_threads(totalThreads/threads) schedule(dynamic,1)
	for (const long sortLength : sortLengths) {
		for (const int sortThreshold : sortThresholds) {
			if (sortThreshold > sortLength) {
				//#pragma omp critical (progress)
				//testCase++;
				continue;
			}
			std::vector<T> sortVector(originalVector.begin(), std::next(originalVector.begin(), sortLength));
		
			auto startTime = std::chrono::steady_clock::now();
			#pragma omp parallel num_threads(threads)
			{
				#pragma omp single
				{
					bitonicSort(sortVector.begin(), sortVector.end(), true, sortThreshold, SortMethod::stdSort);
				}
			}
			auto stopTime = std::chrono::steady_clock::now();
			std::chrono::duration<double> diff = stopTime - startTime;
			if (std::is_sorted(sortVector.begin(), sortVector.end())) {
				MixedSortTimeResult result(SortType::Parallel, threads, sortLength, sortThreshold, SortMethod::Bitonic,
						SortMethod::stdSort, diff, run);
				#pragma omp critical (emplace)
				results.emplace_back(std::make_unique<MixedSortTimeResult>(result));
			}
			
			//#pragma omp critical (progress)
			//printProgress((float) ++testCase / maxTestCasesPerSort); 
		}
	}
	//std::cout << std::endl;
}

template<typename T>
void testSerialMergeInsertionSort(const std::vector<T> &originalVector, 
		const std::vector<long> &sortLengths, 
		const std::vector<int> &sortThresholds, 
		std::vector<std::unique_ptr<SortTimeResult>> &results,
		const int threads,
		const int run)
{
	const int maxTestCasesPerSort = sortLengths.size() * sortThresholds.size();
	//int testCase = 0;
	//std::cout << "testing serial merge insertion sort" << std::endl;
	#pragma omp parallel for schedule(dynamic,1) collapse(2)
	for (const long sortLength : sortLengths) {
		for (const int sortThreshold : sortThresholds) {
			if (sortThreshold > sortLength) {
				//#pragma omp critical (progress)
				//testCase++;
				continue;
			}
			std::vector<T> sortVector(originalVector.begin(), std::next(originalVector.begin(), sortLength));
			
			auto startTime = std::chrono::steady_clock::now();
			serialMergeSort(sortVector.begin(), sortVector.end(), sortThreshold);
			auto stopTime = std::chrono::steady_clock::now();
			std::chrono::duration<double> diff = stopTime - startTime;
			if (std::is_sorted(sortVector.begin(), sortVector.end())) {
				MixedSortTimeResult result(SortType::Serial, 1, sortLength, sortThreshold, SortMethod::Merge,
						SortMethod::Insertion, diff, run);
				#pragma omp critical (emplace)
				results.emplace_back(std::make_unique<MixedSortTimeResult>(result));
			}
			//#pragma omp critical (progress)
			//printProgress((float) ++testCase / maxTestCasesPerSort); 
		}
	}
	//std::cout << std::endl;
}

template<typename T>
void testSerialMergeStdSort(const std::vector<T> &originalVector, 
		const std::vector<long> &sortLengths, 
		const std::vector<int> &sortThresholds, 
		std::vector<std::unique_ptr<SortTimeResult>> &results,
		const int threads,
		const int run)
{
	const int maxTestCasesPerSort = sortLengths.size() * sortThresholds.size();
	//int testCase = 0;
	//std::cout << "testing serial merge with std sort" << std::endl;
	#pragma omp parallel for schedule(dynamic,1) collapse(2)
	for (const long sortLength : sortLengths) {
		for (const int sortThreshold : sortThresholds) {
			if (sortThreshold > sortLength) {
				//#pragma omp critical (progress)
				//testCase++;
				continue;
			}
			std::vector<T> sortVector(originalVector.begin(), std::next(originalVector.begin(), sortLength));
			
			auto startTime = std::chrono::steady_clock::now();
			
			serialMergeSort(sortVector.begin(), sortVector.end(), sortThreshold, SortMethod::stdSort);
			auto stopTime = std::chrono::steady_clock::now();
			std::chrono::duration<double> diff = stopTime - startTime;
			if (std::is_sorted(sortVector.begin(), sortVector.end())) {
				MixedSortTimeResult result(SortType::Serial, 1, sortLength, sortThreshold, SortMethod::Merge,
						SortMethod::stdSort, diff, run);
				#pragma omp critical (emplace)
				results.emplace_back(std::make_unique<MixedSortTimeResult>(result));
			}
					
			//#pragma omp critical (progress)
			//printProgress((float) ++testCase / maxTestCasesPerSort); 
		}
	}
	//std::cout << std::endl;
}

template<typename T>
void testParallelMergeInsertionSort(const std::vector<T> &originalVector, 
		const std::vector<long> &sortLengths, 
		const std::vector<int> &sortThresholds, 
		std::vector<std::unique_ptr<SortTimeResult>> &results,
		const int totalThreads,
		const int threads,
		const int run)
{
	const int maxTestCasesPerSort = sortLengths.size() * sortThresholds.size();
	//int testCase = 0;
	//std::cout << "Testing parallel merge sort with insertion sort and " << threads << " threads" << std::endl;
	//#pragma omp parallel for collapse(2) if(threads<totalThreads) num_threads(totalThreads/threads) schedule(dynamic,1)
	for (const long sortLength : sortLengths) {
		for (const int sortThreshold : sortThresholds) {
			if (sortThreshold > sortLength) {
				//#pragma omp critical (progress)
				//++testCase;
				continue;
			}
			std::vector<T> sortVector(originalVector.begin(), std::next(originalVector.begin(), sortLength));
			
			auto startTime = std::chrono::steady_clock::now();
			#pragma omp parallel num_threads(threads)
			{
				#pragma omp single
				{
					mergeSort(sortVector.begin(), sortVector.end(), sortThreshold);
				}
			}
			auto stopTime = std::chrono::steady_clock::now();
			std::chrono::duration<double> diff = stopTime - startTime;
			if (std::is_sorted(sortVector.begin(), sortVector.end())) {
				MixedSortTimeResult result(SortType::Parallel, threads, sortLength, sortThreshold, SortMethod::Merge,
						SortMethod::Insertion, diff, run);
				#pragma omp critical (emplace)
				results.emplace_back(std::make_unique<MixedSortTimeResult>(result));
			}
			
			//#pragma omp critical (progress)
			//printProgress((float) ++testCase / maxTestCasesPerSort); 
		}
	}
	//std::cout << std::endl;
}

template<typename T>
void testParallelMergeStdSort(const std::vector<T> &originalVector, 
		const std::vector<long> &sortLengths, 
		const std::vector<int> &sortThresholds, 
		std::vector<std::unique_ptr<SortTimeResult>> &results,
		const int totalThreads,
		const int threads,
		const int run)
{
	const int maxTestCasesPerSort = sortLengths.size() * sortThresholds.size();
	//int testCase = 0;
	//std::cout << "Testing parallel merge sort with std::sort and " << threads << " threads" << std::endl;
	//#pragma omp parallel for collapse(2) if(threads<totalThreads) num_threads(totalThreads/threads) schedule(dynamic,1)
	for (const long sortLength : sortLengths) {
		for (const int sortThreshold : sortThresholds) {
			if (sortThreshold > sortLength) {
				//#pragma omp critical (progress)
				//++testCase;
				continue;
			}
			std::vector<T> sortVector(originalVector.begin(), std::next(originalVector.begin(), sortLength));
			
			auto startTime = std::chrono::steady_clock::now();
			#pragma omp parallel num_threads(threads)
			{
				#pragma omp single
				{
					mergeSort(sortVector.begin(), sortVector.end(), sortThreshold, SortMethod::stdSort);
				}
			}
			auto stopTime = std::chrono::steady_clock::now();
			std::chrono::duration<double> diff = stopTime - startTime;
			if (std::is_sorted(sortVector.begin(), sortVector.end())) {
				MixedSortTimeResult result(SortType::Parallel, threads, sortLength, sortThreshold, SortMethod::Merge,
						SortMethod::stdSort, diff, run);
				#pragma omp critical (emplace)
				results.emplace_back(std::make_unique<MixedSortTimeResult>(result));
			}
			
			//#pragma omp critical (progress)
			//printProgress((float) ++testCase / maxTestCasesPerSort);
		}
	}
	//std::cout << std::endl;
}

template<typename T>
void testSerialStdSort(const std::vector<T> &originalVector, 
		const std::vector<long> &sortLengths,
		std::vector<std::unique_ptr<SortTimeResult>> &results,
		const int threads,
		const int run)
{
	//int testCase = 0;
	//std::cout << "testing serial std sort" << std::endl;
	#pragma omp for schedule(dynamic,1)
	for (const long sortLength : sortLengths) {
		std::vector<int> sortVector(originalVector.begin(), std::next(originalVector.begin(), sortLength));
		
		auto startTime = std::chrono::steady_clock::now();
		std::sort(sortVector.begin(), sortVector.end());
		auto stopTime = std::chrono::steady_clock::now();
		std::chrono::duration<double> diff = stopTime - startTime;
		if (std::is_sorted(sortVector.begin(), sortVector.end())) {
			SimpleSortTimeResult result(SortType::Serial, sortLength, SortMethod::stdSort, diff, run);
			#pragma omp critical (emplace)
			results.emplace_back(std::make_unique<SimpleSortTimeResult>(result));
		}
				
		//#pragma omp critical (progress) 
		//printProgress((float) ++testCase / sortLengths.size());
	}
	//std::cout << std::endl;
}

/*
template<typename T>
void testParallelStdSort(const std::vector<T> &originalVector, 
		const std::vector<long> &sortLengths,
		std::vector<std::unique_ptr<SortTimeResult>> &results,
		const int threads,
		const int run)
{
	//int testCase = 0;
	//serial std::sort
	//std::cout << "Testing serial std::sort" << std::endl;
	for (const long sortLength : sortLengths) {
		std::vector<int> sortVector(originalVector.begin(), std::next(originalVector.begin(), sortLength));
		
		auto startTime = std::chrono::steady_clock::now();
		std::sort(std::execution::par, sortVector.begin(), sortVector.end());
		auto stopTime = std::chrono::steady_clock::now();
		std::chrono::duration<double> diff = stopTime - startTime;
		if (std::is_sorted(sortVector.begin(), sortVector.end())) {
			SimpleSortTimeResult result(SortType::Parallel, sortLength, SortMethod::stdSort, diff, run);
			results.emplace_back(std::make_unique<SimpleSortTimeResult>(result));
		}
		//printProgress((float) ++testCase / sortLengths.size()); 
	}
	//std::cout << std::endl;
}
*/

int main(int argc, char** argv) {
	
	if (argc!=2) {
        std::cout << "Usage: sorting_speed " << argv[0] <<
            " [Threads t]"
            << std::endl;
        exit(1);
    }

    int threads = atoi(argv[1]);
	
	std::mt19937 generator(123546);
	
	std::uniform_int_distribution<int> uniformDist(std::numeric_limits<int>::min());
	std::vector<int> sortThresholds = {
		10, 20, 30, 40, 50, 60, 70, 80, 90, 
		100, 200, 300, 400, 500, 600, 700, 800, 900,
		1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000};
	for (int i = 4; i < 15; i++) {
		sortThresholds.push_back(1 << i);
	}
	
	std::vector<long> sortLengths;
	for (int i = 1; i < 6; i++) {
		for (int j = 1; j <= 9; j++) {
			sortLengths.push_back(j * myPow(10, i));
		}
	}
	
	sortLengths.push_back(myPow(10, 6));
	sortLengths.push_back(myPow(10, 7));
	
	for (int i = 1; i < 20; i++) {
		sortLengths.push_back(2 << i);
	}
	std::sort(sortLengths.begin(), sortLengths.end());
	const int maxTestCasesPerSort = sortLengths.size() * sortThresholds.size();
	//length * threshold for bitonic and mixed (serial and parallel) + length for std::sort
	const int maxTestCasesTotal = maxTestCasesPerSort * 8 + sortLengths.size(); 
	
	const long maxLength = sortLengths.back();
	std::vector<int> originalVector(maxLength);
	std::generate(originalVector.begin(), originalVector.end(), [&]() { return uniformDist(generator); });
	
	std::vector<std::unique_ptr<SortTimeResult>> results;
	results.reserve(maxTestCasesTotal);
	
	auto startTotalTime = std::chrono::steady_clock::now();
	
	//for (int run = 1; run <= 3; run++) {
		int run = 1;
		testSerialBitonicStdSort(originalVector, sortLengths, sortThresholds, results, threads, run);
		testSerialBitonicInsertionSort(originalVector, sortLengths, sortThresholds, results, threads, run);
		testSerialMergeStdSort(originalVector, sortLengths, sortThresholds, results, threads, run);
		testSerialMergeInsertionSort(originalVector, sortLengths, sortThresholds, results, threads, run);
		testSerialStdSort(originalVector, sortLengths, results, threads, run);
		for (int i = threads; i > 1; i /= 2) {
			testParallelBitonicStdSort(originalVector, sortLengths, sortThresholds, results, threads, i, run);
			testParallelBitonicInsertionSort(originalVector, sortLengths, sortThresholds, results, threads, i, run);
			testParallelMergeStdSort(originalVector, sortLengths, sortThresholds, results, threads, i, run);
			testParallelMergeInsertionSort(originalVector, sortLengths, sortThresholds, results, threads, i, run);
		}
		//commenting out because intel TBB is erroring on compile
		//testParallelStdSort(originalVector, sortLengths, results, threads, run);
	//}
	/*
	auto stopTotalTime = std::chrono::steady_clock::now();
	std::chrono::duration<double> totalDiff = stopTotalTime - startTotalTime;
	std::cout << "Total time taken for measuring: " << (int) totalDiff.count() / 60 << "m" 
			<< std::fmod(totalDiff.count(), 60) << "s" << std::endl;
			*/

	results.shrink_to_fit();
	std::vector<SortTimeCSVLine> dataCSVVector;
	dataCSVVector.reserve(results.size());
	
	for (const std::unique_ptr<SortTimeResult> &result : results) {
		dataCSVVector.push_back(result->getCSVLine());
	}
	
	//std::ofstream dataFile;
	//dataFile.open("./sorting.csv", std::ios::out );
	//if (!dataFile.is_open()) {
	//	std::cout << "File error" << std::endl;
	//	return 1;
	//}
	
	std::cout << SortTimeCSVLine::getCSVHeader().str();
	for (const SortTimeCSVLine &sortTimeCSVLine : dataCSVVector) {
		std::cout << sortTimeCSVLine.getCSVResultString().str();
	}
	
	std::cout.flush();
	//dataFile.close();
	
	return 0;
}
