import os


def pack_py_to_md(src_dir, output_file="output.md"):
    with open(output_file, "w", encoding="utf-8") as md:
        # 遍历目录下的所有文件
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    md.write(f"{file_path}\n")
                    md.write("```python\n")
                    with open(file_path, "r", encoding="utf-8") as f:
                        md.write(f.read())
                    md.write("\n```\n\n")


if __name__ == "__main__":
    # 修改这里，换成你需要打包的目录
    source_directory = "./src/polar_flow/server"
    pack_py_to_md(source_directory, "all_code.md")
