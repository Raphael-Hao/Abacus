- Compilation
  ```shell
  $ git submodule update --init --recursive
  $ mkdir build && cd build
  $ cmake ..
  $ make -j$(nproc)
  ```
- Run
  ```shell
  $ ./profile_conv ../test.json
  ```