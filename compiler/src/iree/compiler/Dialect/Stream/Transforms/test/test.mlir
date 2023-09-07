func.func @multiTie(%arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index

  %0 = stream.async.dispatch @ex::@dispatch(
         %arg0[%c0 to %c2 for %c2], 
         %arg0[%c2 to %c4 for %c2]
       ) : (!stream.resource<external>{%c4}, 
            !stream.resource<external>{%c4}) -> %arg0{%c4}

  return %0 : !stream.resource<external>
}
