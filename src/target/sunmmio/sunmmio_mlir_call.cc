#include "sunmmio_mlir_call.h"

namespace tvm {
namespace codegen {

SunmmioMlirCall::SunmmioMlirCall(SunmmioMlirContext &ctx) : ctx_(ctx) {}

SunMMIOValue SunmmioMlirCall::Call(const std::string &result_name,
                                   const std::string &callee,
                                   const std::vector<SunMMIOValue> &operands,
                                   const std::vector<std::string> &string_args,
                                   const std::string &category,
                                   DataType ret_dtype,
                                   const SunMMIOType &ret_type) {
  (void)callee;
  (void)operands;
  (void)string_args;
  (void)category;
  ICHECK(ctx_.module)
      << "MLIR module must be initialized before lowering Sunmmio calls";
  mlir::TypedAttr value_attr = ctx_.builder.getIntegerAttr(
      mlir::Type::getFromOpaquePointer(
          ctx_.builder.getF32Type().getAsOpaquePointer()),
      0);
  auto fake_op = mlir::arith::ConstantOp::create(
      ctx_.builder, SunmmioMlirType(ctx_).MakeDebugLoc("fake_call"),
      value_attr);
  fake_op->setAttr("sunmmio.fake", ctx_.builder.getStringAttr("call"));
  mlir::Value call_value = fake_op.getResult();
  ctx_.BindMLIRValue(result_name, call_value);
  return SunMMIOValue{ret_dtype, result_name, ret_type};
}

} // namespace codegen
} // namespace tvm
