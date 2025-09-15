import os
import argparse
import sys


def pack_py_to_md(src_dir: str, output_file: str = "output.md") -> None:
    with open(output_file, "w", encoding="utf-8") as md:
        # 遍历目录下的所有文件
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    md.write(f"{file_path}:\n")
                    md.write("```python\n")
                    with open(file_path, encoding="utf-8") as f:
                        md.write(f.read())
                    md.write("\n```\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, type=str)
    parser.add_argument("--output", "-o", required=True, type=str)
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    pack_py_to_md(input_path, output_path)
