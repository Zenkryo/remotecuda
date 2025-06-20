#!/usr/bin/env python3

import re
import os


def extract_test_cases(file_path):
    """Extract all test cases from the original file"""
    with open(file_path, "r") as f:
        content = f.read()

    # Find all test cases
    test_pattern = r"TEST_F\(CudaRuntimeApiTest, (\S+?)\)(.*?)(?=TEST_F|int main)"
    matches = re.findall(test_pattern, content, re.DOTALL)

    test_cases = []
    for match in matches:
        test_name = match[0]
        test_body = match[1].strip()

        # Remove the empty lines or comments at the end of the test body
        lines = test_body.split("\n")
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == "":
                lines.pop(i)
            elif lines[i].strip().startswith("//"):
                lines.pop(i)
        test_body = "\n".join(lines)

        test_cases.append((test_name, test_body))

    return test_cases


def create_test_file(test_name, test_body, output_dir):
    """Create an individual test file"""
    filename = f"test_{test_name.lower()}.cu"
    filepath = os.path.join(output_dir, filename)
    print("filepath: ", filepath)

    # Convert CamelCase to snake_case for filename
    filename = re.sub(r"(?<!^)(?=[A-Z])", "_", test_name).lower()
    filename = f"test_{filename}.cu"
    filepath = os.path.join(output_dir, filename)

    content = f"""#include "common.h"

TEST_F(CudaRuntimeApiTest, {test_name}){test_body}
"""
    print("content: ", content)
    with open(filepath, "w") as f:
        f.write(content)

    print(f"Created: {filename}")


def main():
    original_file = "../cuda_runtime_api_tests.cu"
    output_dir = "."

    # Extract test cases
    test_cases = extract_test_cases(original_file)

    print(f"Found {len(test_cases)} test cases")

    # Create individual test files
    for test_name, test_body in test_cases:
        filename = re.sub(r"(?<!^)(?=[A-Z])", "_", test_name).lower()
        filename = f"test_{filename}.cu"
        create_test_file(test_name, test_body, output_dir)


if __name__ == "__main__":
    main()
