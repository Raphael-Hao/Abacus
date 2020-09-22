LogSoftmax {#dev_guide_logsoftmax}
============================

>
> [API Reference](@ref dnnl_api_logsoftmax)
>

The logsoftmax primitive performs softmax along a particular axis on data with
arbitrary dimensions followed by the logarithm function. All other axes are
treated as independent (batch).

In general form, the operation is defined by the following formulas (the
variable names follow the standard @ref dev_guide_conventions). Second form is
used as more numerically stable:

### Forward

\f[
    \dst(\overline{ou}, c, \overline{in}) =
        \ln\left({\frac
        {
            e^{\src(\overline{ou}, c, \overline{in}) - \nu(\overline{ou}, \overline{in})}
        }
        {
            \sum\limits_{ic}
                e^{\src(\overline{ou}, ic, \overline{in}) - \nu(\overline{ou}, \overline{in})}
        }}\right) =
        \left(\src(\overline{ou}, c, \overline{in}) - \nu(\overline{ou}, \overline{in})\right)
            - \ln\left(
                    \sum\limits_{ic}
                    e^{\src(\overline{ou}, ic, \overline{in}) - \nu(\overline{ou}, \overline{in})}
                 \right),
\f]

where

- \f$c\f$ axis over which the logsoftmax computation is computed on,
- \f$\overline{ou}\f$ is the outermost index (to the left of logsoftmax axis),
- \f$\overline{in}\f$ is the innermost index (to the right of logsoftmax axis), and
- \f$\nu\f$ is used to produce more accurate results and defined as:

\f[
    \nu(\overline{ou}, \overline{in}) =
        \max\limits_{ic}
        \src(\overline{ou}, ic, \overline{in})
\f]

#### Difference Between Forward Training and Forward Inference

There is no difference between the #dnnl_forward_training
and #dnnl_forward_inference propagation kinds.

### Backward

The backward propagation computes \f$\diffsrc(ou, c, in)\f$, based on
\f$\diffdst(ou, c, in)\f$ and \f$\dst(ou, c, in)\f$.

## Execution Arguments
When executed, the inputs and outputs should be mapped to an execution
argument index as specified by the following table.
| Primitive input/output | Execution argument index |
| ---                    | ---                      |
| \src                   | DNNL_ARG_SRC             |
| \dst                   | DNNL_ARG_DST             |
| \diffsrc               | DNNL_ARG_DIFF_SRC        |
| \diffdst               | DNNL_ARG_DIFF_DST        |

## Implementation Details

### General Notes

1. Both forward and backward propagation support in-place operations, meaning
   that `src` can be used as input and output for forward propagation, and
   `diff_dst` can be used as input and output for backward propagation. In case
   of in-place operation, the original data will be overwritten.

### Post-ops and Attributes

The logsoftmax primitive doesn't support any post-ops or attributes.

### Data Type Support

The logsoftmax primitive supports the following combinations of data types:

| Propagation        | Source / Destination
| :--                | :--
| forward / backward | f32

### Data Representation

#### Source, Destination, and Their Gradients

The logsoftmax primitive works with arbitrary data tensors. There is no special
meaning associated with any logical dimensions. However, the logsoftmax axis is
typically referred to as channels (hence in formulas we use \f$c\f$).


## Implementation Limitations

1. No primitive specific limitations. Refer to @ref dev_guide_data_types for
   limitations related to data types support.

2. **GPU**
    - No support.

## Performance Tips

1. Use in-place operations whenever possible.

2. Currently the softmax primitive is optimized for the cases where
   the dimension of the softmax axis is physically dense. For instance:
   - Optimized: 2D case, tensor \f$A \times B\f$,
                softmax axis 1 (B), format tag #dnnl_ab
   - Optimized: 4D case, tensor \f$A \times B \times C \times D\f$,
                softmax axis 3 (D), format tag #dnnl_abcd
   - Optimized: 4D case, tensor \f$A \times B \times C \times D\f$,
                softmax axis 1 (B), format tag #dnnl_abcd, and
                \f$C = D = 1\f$
   - Optimized: 4D case, tensor \f$A \times B \times C \times D\f$,
                softmax axis 1 (B), format tag #dnnl_acdb or #dnnl_aBcd16b, and
                \f$C \cdot D \ne 1\f$
   - Non-optimized: 2D case, tensor \f$A \times B\f$,
                    softmax axis 0 (A), format tag #dnnl_ab,
                    and \f$B \ne 1\f$
   - Non-optimized: 2D case, tensor \f$A \times B\f$,
                    softmax axis 1 (B), format tag #dnnl_ba,
                    and \f$A \ne 1\f$
   - Non-optimized: 4D case, tensor \f$A \times B \times C \times D\f$,
                    softmax axis 2 (C), format tag #dnnl_acdb, and
                    and \f$D \cdot B \ne 1\f$
