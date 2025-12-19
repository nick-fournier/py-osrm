#include "types/jsoncontainer_nb.h"

#include "util/json_container.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/make_iterator.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;
namespace json = osrm::util::json;

// Forward declaration
nb::object json_value_to_python(const json::Value& value);

// Recursively convert json::Object to Python dict
nb::dict json_object_to_dict(const json::Object& obj) {
    nb::dict result;
    for (const auto& [key, value] : obj.values) {
        result[std::string(key).c_str()] = json_value_to_python(value);
    }
    return result;
}

// Recursively convert json::Array to Python list
nb::list json_array_to_list(const json::Array& arr) {
    nb::list result;
    for (const auto& value : arr.values) {
        result.append(json_value_to_python(value));
    }
    return result;
}

// Convert json::Value to appropriate Python type
nb::object json_value_to_python(const json::Value& value) {
    return std::visit([](auto&& arg) -> nb::object {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, json::String>) {
            return nb::cast(arg.value);
        } else if constexpr (std::is_same_v<T, json::Number>) {
            return nb::cast(arg.value);
        } else if constexpr (std::is_same_v<T, json::Object>) {
            return json_object_to_dict(arg);
        } else if constexpr (std::is_same_v<T, json::Array>) {
            return json_array_to_list(arg);
        } else if constexpr (std::is_same_v<T, json::True>) {
            return nb::cast(true);
        } else if constexpr (std::is_same_v<T, json::False>) {
            return nb::cast(false);
        } else if constexpr (std::is_same_v<T, json::Null>) {
            return nb::none();
        }
    }, value);
}

void init_JSONContainer(nb::module_& m) {
    nb::class_<json::Object>(m, "Object")
        .def(nb::init<>())
        .def("__len__", [](const json::Object& obj) {
            return obj.values.size();
        })
        .def("__bool__", [](const json::Object& obj) {
            return !obj.values.empty();
        })
        .def("__repr__", [](const json::Object& obj) {
            ValueStringifyVisitor visitor;
            return visitor.visitobject(obj);
        })
        .def("__getitem__", [](json::Object& obj, const std::string& key) {
            return obj.values[key];
        })
        .def("__iter__", [](const json::Object& obj) {
            return nb::make_iterator(nb::type<json::Value>(), "iterator",
                                    obj.values.begin(), obj.values.end());
        }, nb::keep_alive<0, 1>())
        .def("to_dict", [](const json::Object& obj) {
            return json_object_to_dict(obj);
        }, "Convert to Python dict");

    nb::class_<json::Array>(m, "Array")
        .def(nb::init<>())
        .def("__len__", [](const json::Array& arr) {
            return arr.values.size();
        })
        .def("__bool__", [](const json::Array& arr) {
            return !arr.values.empty();
        })
        .def("__repr__", [](const json::Array& arr) {
            ValueStringifyVisitor visitor;
            return visitor.visitarray(arr);
        })
        .def("__getitem__", [](json::Array& arr, int i) {
            return arr.values[i];
        })
        .def("__iter__", [](const json::Array& arr) {
            return nb::make_iterator(nb::type<json::Value>(), "iterator",
                                    arr.values.begin(), arr.values.end());
        }, nb::keep_alive<0, 1>())
        .def("to_list", [](const json::Array& arr) {
            return json_array_to_list(arr);
        }, "Convert to Python list");

    nb::class_<json::String>(m, "String")
        .def(nb::init<std::string>());
    nb::class_<json::Number>(m, "Number")
        .def(nb::init<double>());

    nb::class_<json::True>(m, "True")
        .def(nb::init<>());
    nb::class_<json::False>(m, "False")
        .def(nb::init<>());
    nb::class_<json::Null>(m, "Null")
        .def(nb::init<>());
}
