#include "osrm/extractor.hpp"
#include "osrm/contractor.hpp"
#include "osrm/partitioner.hpp"
#include "osrm/customizer.hpp"
#include "osrm/extractor_config.hpp"
#include "osrm/contractor_config.hpp"
#include "osrm/partitioner_config.hpp"
#include "osrm/customizer_config.hpp"

#include "util/log.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/function.h>

#include <sstream>
#include <chrono>
#include <iostream>
#include <thread>

#include "extractorconfig_nb.h"
#include "contractorconfig_nb.h"
#include "partitionerconfig_nb.h"
#include "customizerconfig_nb.h"

namespace nb = nanobind;

// Helper class to capture stdout/stderr and optionally call progress callback
class OutputCapture {
private:
    std::stringstream stdout_buffer;
    std::stringstream stderr_buffer;
    nb::object callback;
    
    std::streambuf* old_stdout = nullptr;
    std::streambuf* old_stderr = nullptr;
    
    bool capturing = false;

public:
    OutputCapture(nb::object cb = nb::none()) 
        : callback(cb) {}
    
    void start() {
        if (capturing) return;
        
        old_stdout = std::cout.rdbuf();
        old_stderr = std::cerr.rdbuf();
        
        std::cout.rdbuf(stdout_buffer.rdbuf());
        std::cerr.rdbuf(stderr_buffer.rdbuf());
        
        capturing = true;
    }
    
    void stop() {
        if (!capturing) return;
        
        if (old_stdout) {
            std::cout.rdbuf(old_stdout);
            std::cerr.rdbuf(old_stderr);
            old_stdout = nullptr;
            old_stderr = nullptr;
        }
        
        // Process callbacks for captured output
        if (!callback.is_none()) {
            process_captured_output();
        }
        
        capturing = false;
    }
    
    void process_captured_output() {
        nb::gil_scoped_acquire guard;
        
        try {
            // Process stdout line by line
            std::string line;
            stdout_buffer.clear();
            stdout_buffer.seekg(0);
            while (std::getline(stdout_buffer, line)) {
                if (!line.empty()) {
                    callback(line);
                }
            }
            
            // Process stderr line by line
            stderr_buffer.clear();
            stderr_buffer.seekg(0);
            while (std::getline(stderr_buffer, line)) {
                if (!line.empty()) {
                    callback(line);
                }
            }
        } catch (...) {
            // Ignore callback errors to prevent crashing
        }
    }
    
    std::string get_stdout() const {
        return stdout_buffer.str();
    }
    
    std::string get_stderr() const {
        return stderr_buffer.str();
    }
    
    ~OutputCapture() {
        stop();
    }
};

// Helper function to set log level from string
void set_log_level(const std::string& verbosity) {
    using osrm::util::LogPolicy;
    
    LogPolicy::GetInstance().Unmute();
    
    if (verbosity == "NONE") {
        LogPolicy::GetInstance().Mute();
    } else if (verbosity == "ERROR") {
        LogPolicy::GetInstance().SetLevel(logERROR);
    } else if (verbosity == "WARNING") {
        LogPolicy::GetInstance().SetLevel(logWARNING);
    } else if (verbosity == "INFO") {
        LogPolicy::GetInstance().SetLevel(logINFO);
    } else if (verbosity == "DEBUG") {
        LogPolicy::GetInstance().SetLevel(logDEBUG);
    } else {
        // Default to INFO
        LogPolicy::GetInstance().SetLevel(logINFO);
    }
}

// Extract with output capture and optional progress callback
nb::dict extract_with_capture(
    const osrm::extractor::ExtractorConfig& config,
    const std::string& verbosity,
    nb::object progress_callback = nb::none()
) {
    OutputCapture capture(progress_callback);
    
    auto start = std::chrono::steady_clock::now();
    
    set_log_level(verbosity);
    
    // Start capture
    capture.start();
    
    bool success = true;
    std::string error_msg;
    
    try {
        // Release GIL for long-running C++ operation
        nb::gil_scoped_release release;
        osrm::extract(config);
    } catch (const std::exception& e) {
        success = false;
        error_msg = e.what();
    } catch (...) {
        success = false;
        error_msg = "Unknown error during extraction";
    }
    
    capture.stop();
    
    auto end = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();
    
    nb::dict result;
    result["success"] = success;
    result["duration"] = duration;
    result["stdout"] = capture.get_stdout();
    result["stderr"] = capture.get_stderr();
    if (!success) {
        result["error"] = error_msg;
    }
    
    return result;
}

// Simple extract without capture (output goes directly to stdout/stderr)
void extract_simple(
    const osrm::extractor::ExtractorConfig& config,
    const std::string& verbosity
) {
    set_log_level(verbosity);
    
    nb::gil_scoped_release release;
    osrm::extract(config);
}

// Contract with output capture and optional progress callback
nb::dict contract_with_capture(
    const osrm::contractor::ContractorConfig& config,
    const std::string& verbosity,
    nb::object progress_callback = nb::none()
) {
    OutputCapture capture(progress_callback);
    
    auto start = std::chrono::steady_clock::now();
    
    set_log_level(verbosity);
    
    capture.start();
    
    bool success = true;
    std::string error_msg;
    
    try {
        nb::gil_scoped_release release;
        osrm::contract(config);
    } catch (const std::exception& e) {
        success = false;
        error_msg = e.what();
    } catch (...) {
        success = false;
        error_msg = "Unknown error during contraction";
    }
    
    capture.stop();
    
    auto end = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();
    
    nb::dict result;
    result["success"] = success;
    result["duration"] = duration;
    result["stdout"] = capture.get_stdout();
    result["stderr"] = capture.get_stderr();
    if (!success) {
        result["error"] = error_msg;
    }
    
    return result;
}

// Simple contract without capture
void contract_simple(
    const osrm::contractor::ContractorConfig& config,
    const std::string& verbosity
) {
    set_log_level(verbosity);
    
    nb::gil_scoped_release release;
    osrm::contract(config);
}

// Partition with output capture and optional progress callback
nb::dict partition_with_capture(
    const osrm::partitioner::PartitionerConfig& config,
    const std::string& verbosity,
    nb::object progress_callback = nb::none()
) {
    OutputCapture capture(progress_callback);
    
    auto start = std::chrono::steady_clock::now();
    
    set_log_level(verbosity);
    
    capture.start();
    
    bool success = true;
    std::string error_msg;
    
    try {
        nb::gil_scoped_release release;
        osrm::partition(config);
    } catch (const std::exception& e) {
        success = false;
        error_msg = e.what();
    } catch (...) {
        success = false;
        error_msg = "Unknown error during partitioning";
    }
    
    capture.stop();
    
    auto end = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();
    
    nb::dict result;
    result["success"] = success;
    result["duration"] = duration;
    result["stdout"] = capture.get_stdout();
    result["stderr"] = capture.get_stderr();
    if (!success) {
        result["error"] = error_msg;
    }
    
    return result;
}

// Simple partition without capture
void partition_simple(
    const osrm::partitioner::PartitionerConfig& config,
    const std::string& verbosity
) {
    set_log_level(verbosity);
    
    nb::gil_scoped_release release;
    osrm::partition(config);
}

// Customize with output capture and optional progress callback
nb::dict customize_with_capture(
    const osrm::customizer::CustomizationConfig& config,
    const std::string& verbosity,
    nb::object progress_callback = nb::none()
) {
    OutputCapture capture(progress_callback);
    
    auto start = std::chrono::steady_clock::now();
    
    set_log_level(verbosity);
    
    capture.start();
    
    bool success = true;
    std::string error_msg;
    
    try {
        nb::gil_scoped_release release;
        osrm::customize(config);
    } catch (const std::exception& e) {
        success = false;
        error_msg = e.what();
    } catch (...) {
        success = false;
        error_msg = "Unknown error during customization";
    }
    
    capture.stop();
    
    auto end = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();
    
    nb::dict result;
    result["success"] = success;
    result["duration"] = duration;
    result["stdout"] = capture.get_stdout();
    result["stderr"] = capture.get_stderr();
    if (!success) {
        result["error"] = error_msg;
    }
    
    return result;
}

// Simple customize without capture
void customize_simple(
    const osrm::customizer::CustomizationConfig& config,
    const std::string& verbosity
) {
    set_log_level(verbosity);
    
    nb::gil_scoped_release release;
    osrm::customize(config);
}

void init_Preprocessing(nb::module_& m) {
    // Config classes
    init_ExtractorConfig(m);
    init_ContractorConfig(m);
    init_PartitionerConfig(m);
    init_CustomizerConfig(m);
    
    // Simple preprocessing functions (output to stdout/stderr)
    m.def("extract", &extract_simple,
          nb::arg("config"),
          nb::arg("verbosity") = "INFO",
          "Extract OSM data (output to stdout/stderr)");
    
    m.def("contract", &contract_simple,
          nb::arg("config"),
          nb::arg("verbosity") = "INFO",
          "Contract graph for CH algorithm (output to stdout/stderr)");
    
    m.def("partition", &partition_simple,
          nb::arg("config"),
          nb::arg("verbosity") = "INFO",
          "Partition graph for MLD algorithm (output to stdout/stderr)");
    
    m.def("customize", &customize_simple,
          nb::arg("config"),
          nb::arg("verbosity") = "INFO",
          "Customize partitioned graph for MLD (output to stdout/stderr)");
    
    // Preprocessing functions with output capture
    m.def("extract_with_capture", &extract_with_capture,
          nb::arg("config"),
          nb::arg("verbosity") = "INFO",
          nb::arg("progress_callback") = nb::none(),
          "Extract OSM data with output capture and optional progress callback");
    
    m.def("contract_with_capture", &contract_with_capture,
          nb::arg("config"),
          nb::arg("verbosity") = "INFO",
          nb::arg("progress_callback") = nb::none(),
          "Contract graph with output capture and optional progress callback");
    
    m.def("partition_with_capture", &partition_with_capture,
          nb::arg("config"),
          nb::arg("verbosity") = "INFO",
          nb::arg("progress_callback") = nb::none(),
          "Partition graph with output capture and optional progress callback");
    
    m.def("customize_with_capture", &customize_with_capture,
          nb::arg("config"),
          nb::arg("verbosity") = "INFO",
          nb::arg("progress_callback") = nb::none(),
          "Customize graph with output capture and optional progress callback");
}
