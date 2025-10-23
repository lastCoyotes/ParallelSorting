#include <map>
#include <string>
#include <string_view>
#include <sstream>
#include <iomanip>
#include <chrono>

enum class SortType {
	Serial,
	Parallel
};

enum class SortMethod {
	None,
	Bitonic,
	Merge,
	Insertion,
	stdSort
};

const std::map<const SortType, const std::string_view> typeStringMap = { 
	{SortType::Serial, "serial"},
	{SortType::Parallel, "parallel"}
};

const std::map<const SortMethod, const std::string_view> methodStringMap = {
	{SortMethod::Bitonic, "Bitonic"},
	{SortMethod::Merge, "Merge"},
	{SortMethod::Insertion, "Insertion"},
	{SortMethod::stdSort, "std::sort"}
};

class SortTimeCSVLine {
protected:
	const long sortLength;
	const long sortThreshold;
	const SortType type;
	const int threads;
	const SortMethod outerOrOnlyMethod;
	const SortMethod innerMethod;
	const std::chrono::duration<double> sortTime;
	const int run;
	
public:

	SortTimeCSVLine(const long sortLength, const long sortThreshold, const SortType type, const int threads,
			const SortMethod outerMethod, const SortMethod innerMethod, const std::chrono::duration<double> sortTime,
			const int run) :
		sortLength(sortLength),
		sortThreshold(sortThreshold),
		type(type),
		threads(threads),
		outerOrOnlyMethod(outerMethod),
		innerMethod(innerMethod),
		sortTime(sortTime),
		run(run) {};

	std::stringstream getCSVResultString() const {
		std::stringstream output;
		output << typeStringMap.at(type) << ","
				<< threads << ","
				<< methodStringMap.at(outerOrOnlyMethod) << ","
				<< ((innerMethod != SortMethod::None) ? methodStringMap.at(innerMethod) : "") << ","
				<< sortLength << ","
				<< ((sortThreshold >= 0) ? std::to_string(sortThreshold) : "") << ","
				<< sortTime.count() << ","
				<< run << "\n";
		return output;
	};
	
	static std::stringstream getCSVHeader() {
		std::stringstream output;
		output << "SortType,Threads,OuterOrOnlyMethod,InnerMethod,SortLength,SortThreshold,SortTime,Run\n";
		return output;
	}
};

class SortTimeResult {
public:
	const long sortLength;
	const SortType type;
	const std::chrono::duration<double> sortTime;
	const int run;

	SortTimeResult(const long sortLength, const SortType type,
			const std::chrono::duration<double> &sortTime, const int run):
			sortLength(sortLength),
			type(type),
			sortTime(sortTime),
			run(run) {};
			
	virtual std::stringstream getResultString() const = 0;
	virtual SortTimeCSVLine getCSVLine() const = 0;
	//virtual ~SortTimeResult() {};
};

class MixedSortTimeResult : public virtual SortTimeResult {
public:
	const long sortThreshold;
	const SortMethod outerMethod;
	const SortMethod innerMethod;
	const int threads;

	MixedSortTimeResult(const SortType type,  const int threads, const long sortLength, const long sortThreshold,
			const SortMethod outerMethod, const SortMethod innerMethod,
			const std::chrono::duration<double> &sortTime, const int run):
			SortTimeResult{sortLength, type, sortTime, run},
			sortThreshold(sortThreshold),
			outerMethod(outerMethod),
			innerMethod(innerMethod),
			threads(threads){};
			
	std::stringstream getResultString() const override{
		std::stringstream output;
		output << "Time to sort " << std::setw(9) << sortLength << " elements with "
				<< typeStringMap.at(type) << " " << methodStringMap.at(outerMethod)
				<< " sort with threshold " << std::setw(6) << sortThreshold << " for "
				<< methodStringMap.at(innerMethod) << " sort: " << sortTime.count();
		return output;
	};
	
	SortTimeCSVLine getCSVLine() const override;
	//~MixedSortTimeResult() {};
};

class SimpleSortTimeResult : public virtual SortTimeResult {
public:
	const SortMethod method;
	
	SimpleSortTimeResult(const SortType type, const long sortLength,
			const SortMethod method,
			const std::chrono::duration<double> &sortTime, const int run):
			SortTimeResult{sortLength, type, sortTime, run},
			method(method) {};
			
	std::stringstream getResultString() const override {
		std::stringstream output;
		output << "Time to sort " << std::setw(9) << sortLength << " elements with "
				<< typeStringMap.at(type) << " " << methodStringMap.at(method) << " sort: " << sortTime.count();
		return output;
	};
	
	SortTimeCSVLine getCSVLine() const override;
	//~SimpleSortTimeResult() {};
};

class SortTimeCSVLineFactory {
public:
	SortTimeCSVLine makeCSVLineFromMixedSort(const MixedSortTimeResult &result) const {
		return SortTimeCSVLine(result.sortLength,
				result.sortThreshold,
				result.type,
				result.threads,
				result.outerMethod,
				result.innerMethod,
				result.sortTime,
				result.run);
	}
	
	SortTimeCSVLine makeCSVLineFromSimpleSort(const SimpleSortTimeResult &result) const {
		return SortTimeCSVLine(result.sortLength,
				-1,
				result.type,
				1,
				result.method,
				SortMethod::None,
				result.sortTime,
				result.run);
	}
};

const SortTimeCSVLineFactory factory;

SortTimeCSVLine MixedSortTimeResult::getCSVLine() const {
	return factory.makeCSVLineFromMixedSort(*this);
}

SortTimeCSVLine SimpleSortTimeResult::getCSVLine() const {
	return factory.makeCSVLineFromSimpleSort(*this);
}