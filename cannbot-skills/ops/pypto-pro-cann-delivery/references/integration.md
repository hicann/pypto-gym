# PyPTO-Pro 接入与验收细节

在主流程步骤 3 生成交付件前读取。以下补充 PyPTO-Pro 与 CANN 的连接要求，不替代官方模板或精度 Skill；接口名称、签名和布局须与当前版本源码及真实生成结果核对。GEIR 专属要求仅在选择 GEIR 时执行。

## 参数与注册

### CPU 与 kernel 的连接

- 将 wrapper 中仅依赖公开输入、属性和芯片信息的切分计算迁到 host，保持 kernel 数学计算不变。host 同时设置启动核数（blockDim）和临时内存（workspace）。
- 运行时值放入 Python `@dataclass` TilingData。host 对 `context->GetTilingData<生成类型>()` 判空后按生成布局填写，字段、类型和定长数组与 kernel 一致。
- host 另定义仅供注册和元数据使用的 TilingData 类，其字段、类型和定长数组须与生成布局一致；以准确 OpType 执行 `REGISTER_TILING_DATA_CLASS`，不得把该注册类当普通 buffer 写入。
- 定义并注册当前 host 接口要求的 CompileInfo 和 TilingParse；无编译期内容时也提供返回成功的空解析，不仅注册 `.Tiling(...)`。
- 用 `@pl.jit(tiling_key=...)` 绑定 TilingKey，改造后的 JIT 测试按官方语法显式传入合法 key。host 按字段声明顺序，将合法候选的实际值传给生成的 `GET_TPL_TILING_KEY(...)`，再 `context->SetTilingKey(...)`，不手工编码。
- 原 kernel 无 TilingData/TilingKey 时，仅在公开范围内无编译期分支且 kernel 不依赖 host 派生值的情况下，使用含保留字段的最小 TilingData 和单候选 key；否则按实际运行数据和分支建模。
- kernel 下发参数按 `inputs -> outputs -> workspace -> tiling` 排列；输入输出共用地址也保留各自参数槽。若动态 `pl.Tensor` 额外展开了下发端没有的 shape/stride 参数，改用 `pl.Ptr`，将形状/步长写入 TilingData，再用 `pl.make_tensor` 重建。

### 名称与工程接入

- `op_kernel/<op_file>.py` 文件名、CMake `<op_file>`、显式 `opFile.value` 和唯一 `@pl.jit` 函数名一致。省略 `opFile.value` 时，以生成的 ops-info 证明默认映射正确。
- 按已读取的真实签名，在 `op_host/CMakeLists.txt` 调用一次 `enable_pypto_kernel(<op_file>)` 或等价接口；将 OpDef 源加入实际 host-def 目标、kernel 加入真实编译集合，不能只有文件或空目标。
- OpDef 为目标 SoC 注册 AICore 配置；动态 rank/shape、类型和布局均与公开契约一致。
- wrapper 生成落点与算子是否支持动态形状无必然关系。若当前官方 kernel target 固定从 `dynamic/` 读取，只在副本的 PyPTO codegen 阶段生成根目录和 `dynamic/` 两份同 SHA-256 wrapper；不在单算子 CMake/configure 时复制尚未生成的文件，也不改变普通 AscendC 路径。
- 空 Tensor 等框架快捷路径可能跳过 host/kernel。若负例证明约束未被检查，将检查前移到实际经过的 API 或图推导入口；不改 kernel 算法。

## 构建与入包

### JIT 对比与配置

用 `importlib` 加载含 dataclass 的文件时，先将 module 注册到 `sys.modules` 再执行。

clean configure 显式给出目标算子、SoC、构建类型、版本和当前打包开关。核对外层和内层构建实际收到同一组值，不只查看外层命令。确认共享头来自本次 codegen、最终 kernel target 可达；仅 ACLNN 时，不意外纳入独立 GEIR/infer 源。之后使用完整 package 入口的增量/复用选项沿用该配置。

### host 真实执行

编译日志必须包含本算子的 `*_tiling.cpp`。加载实际承载它的 host 共享库，从正式注册表取得目标 OpType 的 tiling 函数和非空布局，再执行：

- 复用模板的 context faker、case executor、注册 main 和 CMake 编译/链接设置（含 C++ ABI）；先注册目标原型再构造 context。
- 若测试框架不可用，可采用最小断言程序，但保留 case executor 的构造步骤、执行期间有效的非空 compile/platform info；解引用前检查 `Build()` 结果。
- 合法用例逐字节核对 TilingData，并检查 TilingKey、blockDim、workspace；属于 host 职责的非法输入须被拒绝。
- 注册布局容量覆盖生成布局。包内各份元数据的 `opParaSize` 相互一致且不小于有效字节，不要求与有效字节相等。

### 包完整性

使用已核实的完整打包入口和目标算子筛选；支持时采用有界并行度。核对可安装 `.run` 而非只有 `.run.json` 等描述文件；包内 ops-info 的 OpType/`opFile` 正确，kernel 元数据中的 `opParaSize` 符合上述布局。

将公开范围内 host 可选的 TilingKey、数据类型和布局组合，与实际入包的二进制逐项对应；生成头中的候选不等于已入包。真机用例须覆盖各可达编译分支并关联实际选中的 key，用例设计仍遵循精度 Skill；缺项不得声明全范围通过。

命令返回 0 仍须检查完整日志：针对目标算子的错误、未注册、无 operator info、回退或编译失败均不通过。无目标名的通用提示需结合所属生成分支、最终公开头、导出符号和 ops-info 判断，不凭关键词单独放行或判错。

目标算子的交付文件必须来自本轮安装包；允许使用已确认的 CANN/系统基线库，不通过包外 `LD_LIBRARY_PATH` 指向 build 或旧算子产物补漏。运行前后，安装前缀文件清单与 SHA-256 必须不变。

## 真机与离线证据

### 调用与数据

ACLNN 用例按模板执行 `GetWorkspaceSize` 后的第二段调用、同步、取回全部输出并释放资源。设备号来自基线环境或运行参数，不写死、不调用 `aclrtResetDevice`。第二段已消费的一次性 executor 不再次销毁。

C++ 用例负责真实调用和原始输入输出落盘；正确结果、容差和判定由精度 Skill 指定的进程外检查器完成。非宿主原生类型使用该 Skill 的原始字节方案或 CANN 公共转换 API，不手写编解码。

两条通路中可能导致进程退出的接口负例均逐例隔离，记录返回码或信号，不能遮蔽后续用例。

### GEIR 图调用与隔离

执行 `GEInitialize -> AddGraph -> RunGraph -> GEFinalize`，在 Session options 设置 `ge.jit_compile=0`，不预填目标输出描述；用返回的形状、类型验证原型和 inferShape/inferDtype。非法输入在 `AddGraph` 成功时继续到 `RunGraph`，须在目标 kernel 启动前明确拒绝。

为本轮建立独立 OPP 根：

- vendor 由本轮包实际安装；必要基线资源可复制或以符号链接只读引用，不使用硬链接。
- 按当前 GE 的实际库搜索路径、注册优先级检查布局；目录存在或 `vendors/config.ini` 有记录不足以证明加载正确。兼容映射仅在该根内建立，指向本轮安装文件或已确认的只读基线。
- 设置 `ASCEND_OPP_PATH`，取消指向同一 vendor 的 `ASCEND_CUSTOM_OPP_PATH`，避免重复注册。运行器不得直接链接、预加载或 `dlopen` 自定义 proto。
- 从 OPP/源码根外的新工作目录运行，设置 `PYTHONDONTWRITEBYTECODE=1`。正常用例可批量运行，但每例保留独立输入、输出、结果及调用标识/时间。
- 本次运行前后，隔离 OPP 根及其来源安装前缀的文件清单和 SHA-256 均不变。

### 证明执行的是本轮离线二进制

启动时记录实际进程/任务标识与时间范围，以官方 profiler/runtime trace 或 CANN 日志关联本次成功调用的目标设备执行事件。进程中的线程标识须关联到实际调用进程，不能仅取历史日志“最新命中”。

将目标 `.o` 的实际路径、SHA-256、ops-info/二进制 JSON 与同次 launch/执行事件对应。文件读取只能证明二进制被读取，不能替代目标执行证据；算子名、文件存在、编译成功、退出码 0 或数值正确也不能替代。

GEIR 注册过程读取生成的 Python wrapper 不等于在线编译；须同时证明复用本轮 binary、未读取目标 PyPTO DSL kernel、未调用在线编译工具链、未在本轮 OPP 根外生成新的目标 `.o`/JSON。证据不足时保留已证实结果，不能判定离线交付通过。
