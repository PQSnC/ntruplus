.global poly_basemul
.global _poly_basemul
poly_basemul:
_poly_basemul:
    dst       .req x0
    src1      .req x1
    src2      .req x2
    zetas_ptr     .req x3
    counter   .req x8

    adr zetas_ptr, zetas_mul

    ld1  {v0.8h}, [zetas_ptr], #16

    mov counter, #1728

_looptop:
    #load
    ld1  {v1.8h}, [zetas_ptr], #16

    #load
    ld1 {v4.8h - v6.8h}, [src1], #48
    ld1 {v7.8h - v9.8h}, [src2], #48

    smull  v10.4s, v5.4h, v9.4h //a1*b2
    smull2 v11.4s, v5.8h, v9.8h //a1*b2
    smull  v12.4s, v6.4h, v9.4h //a2*b2
    smull2 v13.4s, v6.8h, v9.8h //a2*b2

    smlal  v10.4s, v6.4h, v8.4h //a2*b1
    smlal2 v11.4s, v6.8h, v8.8h //a2*b1

    uzp1 v15.8h, v12.8h, v13.8h
    uzp1 v14.8h, v10.8h, v11.8h

    mul v15.8h, v15.8h, v0.h[1]
    mul v14.8h, v14.8h, v0.h[1]

    smlal  v12.4s, v15.4h, v0.h[0]
    smlal2 v13.4s, v15.8h, v0.h[0]
    smlal  v10.4s, v14.4h, v0.h[0]
    smlal2 v11.4s, v14.8h, v0.h[0]

    uzp2 v30.8h, v12.8h, v13.8h
    uzp2 v29.8h, v10.8h, v11.8h

    smull  v14.4s, v6.4h, v7.4h //a2*b0
    smull2 v15.4s, v6.8h, v7.8h //a2*b0
    smull  v12.4s, v30.4h, v1.4h //r[1]*zeta
    smull2 v13.4s, v30.8h, v1.8h //r[1]*zeta
    smull  v10.4s, v29.4h, v1.4h //r[0]*zeta
    smull2 v11.4s, v29.8h, v1.8h //r[0]*zeta

    smlal  v14.4s, v4.4h, v9.4h //a0*b2
    smlal2 v15.4s, v4.8h, v9.8h //a0*b2
    smlal  v12.4s, v4.4h, v8.4h //a0*b1
    smlal2 v13.4s, v4.8h, v8.8h //a0*b1
    smlal  v10.4s, v4.4h, v7.4h //a0*b0
    smlal2 v11.4s, v4.8h, v7.8h //a0*b0

    smlal  v14.4s, v5.4h, v8.4h //a1*b1
    smlal2 v15.4s, v5.8h, v8.8h //a1*b1
    smlal  v12.4s, v5.4h, v7.4h //a1*b0
    smlal2 v13.4s, v5.8h, v7.8h //a1*b0

    uzp1 v18.8h, v14.8h, v15.8h
    uzp1 v17.8h, v12.8h, v13.8h
    uzp1 v16.8h, v10.8h, v11.8h

    mul v18.8h, v18.8h, v0.h[1]
    mul v17.8h, v17.8h, v0.h[1]
    mul v16.8h, v16.8h, v0.h[1]

    smlal  v14.4s, v18.4h, v0.h[0]
    smlal2 v15.4s, v18.8h, v0.h[0]
    smlal  v12.4s, v17.4h, v0.h[0]
    smlal2 v13.4s, v17.8h, v0.h[0]
    smlal  v10.4s, v16.4h, v0.h[0]
    smlal2 v11.4s, v16.8h, v0.h[0]

    uzp2 v6.8h, v14.8h, v15.8h
    uzp2 v5.8h, v12.8h, v13.8h
    uzp2 v4.8h, v10.8h, v11.8h

    #store
    st1 {v4.8h - v6.8h}, [dst], #48
    
    subs counter, counter, #48
    b.ne _looptop

    .unreq    dst
    .unreq    src1
    .unreq    src2
    .unreq    zetas_ptr
    .unreq    counter

    ret


.global poly_basemul_add
.global _poly_basemul_add
poly_basemul_add:
_poly_basemul_add:
    dst       .req x0
    src1      .req x1
    src2      .req x2
    src3      .req x3    
    zetas_ptr .req x4
    counter   .req x8

    adr zetas_ptr, zetas_mul

    ld1  {v0.8h}, [zetas_ptr], #16

    mov counter, #1728

_looptop_add:
    #load
    ld1  {v1.8h}, [zetas_ptr], #16

    #load
    ld1 {v4.8h - v6.8h},   [src1], #48
    ld1 {v7.8h - v9.8h},   [src2], #48
    ld1 {v10.8h - v12.8h}, [src3], #48

    smull  v13.4s, v5.4h, v9.4h //a1*b2
    smull2 v14.4s, v5.8h, v9.8h //a1*b2
    smull  v15.4s, v6.4h, v9.4h //a2*b2
    smull2 v16.4s, v6.8h, v9.8h //a2*b2

    smlal  v13.4s, v6.4h, v8.4h //a2*b1
    smlal2 v14.4s, v6.8h, v8.8h //a2*b1

    uzp1 v18.8h, v15.8h, v16.8h
    uzp1 v17.8h, v13.8h, v14.8h

    mul v18.8h, v18.8h, v0.h[1]
    mul v17.8h, v17.8h, v0.h[1]

    smlal  v15.4s, v18.4h, v0.h[0]
    smlal2 v16.4s, v18.8h, v0.h[0]
    smlal  v13.4s, v17.4h, v0.h[0]
    smlal2 v14.4s, v17.8h, v0.h[0]

    uzp2 v30.8h, v15.8h, v16.8h
    uzp2 v29.8h, v13.8h, v14.8h


    smull  v17.4s, v6.4h, v7.4h //a2*b0
    smull2 v18.4s, v6.8h, v7.8h //a2*b0
    smull  v15.4s, v30.4h, v1.4h //r[1]*zeta
    smull2 v16.4s, v30.8h, v1.8h //r[1]*zeta
    smull  v13.4s, v29.4h, v1.4h //r[0]*zeta
    smull2 v14.4s, v29.8h, v1.8h //r[0]*zeta

    smlal  v17.4s, v4.4h, v9.4h //a0*b2
    smlal2 v18.4s, v4.8h, v9.8h //a0*b2
    smlal  v15.4s, v4.4h, v8.4h //a0*b1
    smlal2 v16.4s, v4.8h, v8.8h //a0*b1
    smlal  v13.4s, v4.4h, v7.4h //a0*b0
    smlal2 v14.4s, v4.8h, v7.8h //a0*b0

    smlal  v17.4s, v5.4h, v8.4h //a1*b1
    smlal2 v18.4s, v5.8h, v8.8h //a1*b1
    smlal  v15.4s, v5.4h, v7.4h //a1*b0
    smlal2 v16.4s, v5.8h, v7.8h //a1*b0

    smlal  v13.4s, v10.4h, v0.h[2] //c0*R
    smlal2 v14.4s, v10.8h, v0.h[2] //c0*R
    smlal  v15.4s, v11.4h, v0.h[2] //c1*R
    smlal2 v16.4s, v11.8h, v0.h[2] //c1*R
    smlal  v17.4s, v12.4h, v0.h[2] //c1*R
    smlal2 v18.4s, v12.8h, v0.h[2] //c1*R

    uzp1 v19.8h, v13.8h, v14.8h
    uzp1 v20.8h, v15.8h, v16.8h
    uzp1 v21.8h, v17.8h, v18.8h

    mul v19.8h, v19.8h, v0.h[1]
    mul v20.8h, v20.8h, v0.h[1]
    mul v21.8h, v21.8h, v0.h[1]

    smlal  v13.4s, v19.4h, v0.h[0]
    smlal2 v14.4s, v19.8h, v0.h[0]
    smlal  v15.4s, v20.4h, v0.h[0]
    smlal2 v16.4s, v20.8h, v0.h[0]
    smlal  v17.4s, v21.4h, v0.h[0]
    smlal2 v18.4s, v21.8h, v0.h[0]

    uzp2 v4.8h, v13.8h, v14.8h
    uzp2 v5.8h, v15.8h, v16.8h
    uzp2 v6.8h, v17.8h, v18.8h

    #store
    #store
    st1 {v4.8h - v6.8h}, [dst], #48
    
    subs counter, counter, #48
    b.ne _looptop_add

    .unreq    dst
    .unreq    src1
    .unreq    src2
    .unreq    src3
    .unreq    zetas_ptr
    .unreq    counter

    ret


.global poly_baseinv
.global _poly_baseinv
poly_baseinv:
_poly_baseinv:
    dst       .req x0
    src       .req x1
    zetas_ptr .req x2
    counter1  .req x8
    counter2  .req x9
    stack_ptr .req x10
    
    adr zetas_ptr, zetas_mul

    ld1  {v0.8h}, [zetas_ptr], #16

    sub sp, sp, #96

    mov counter1, #1728

_looptop_baseinv:

    mov counter2, #288

    mov stack_ptr, sp

_loop_inner1:
    #load
    ld1 {v1.8h, v2.8h}, [zetas_ptr], #32

    #load
    ld1 {v3.8h, v4.8h, v5.8h}, [src], #48
    ld1 {v6.8h, v7.8h, v8.8h}, [src], #48

    neg v30.8h, v3.8h //-a0
    neg v31.8h, v6.8h //-a0

    smull   v9.4s, v4.4h, v5.4h //a1*a2
    smull2 v10.4s, v4.8h, v5.8h //a1*a2
    smull  v11.4s, v7.4h, v8.4h //a1*a2
    smull2 v12.4s, v7.8h, v8.8h //a1*a2

    smull  v13.4s, v5.4h, v5.4h //a2*a2
    smull2 v14.4s, v5.8h, v5.8h //a2*a2
    smull  v15.4s, v8.4h, v8.4h //a2*a2
    smull2 v16.4s, v8.8h, v8.8h //a2*a2

    smull  v17.4s, v4.4h, v4.4h //a1*a1
    smull2 v18.4s, v4.8h, v4.8h //a1*a1
    smull  v19.4s, v7.4h, v7.4h //a1*a1
    smull2 v20.4s, v7.8h, v7.8h //a1*a1

    smlal  v17.4s, v30.4h, v5.4h //-a0*a2
    smlal2 v18.4s, v30.8h, v5.8h //-a0*a2
    smlal  v19.4s, v31.4h, v8.4h //-a0*a2
    smlal2 v20.4s, v31.8h, v8.8h //-a0*a2

    uzp1 v24.8h,  v9.8h, v10.8h
    uzp1 v25.8h, v11.8h, v12.8h
    uzp1 v26.8h, v13.8h, v14.8h
    uzp1 v27.8h, v15.8h, v16.8h
    uzp1 v28.8h, v17.8h, v18.8h
    uzp1 v29.8h, v19.8h, v20.8h

    mul v24.8h, v24.8h, v0.h[1] //r[0]
    mul v25.8h, v25.8h, v0.h[1] //r[0]
    mul v26.8h, v26.8h, v0.h[1] //r[1]
    mul v27.8h, v27.8h, v0.h[1] //r[1]
    mul v28.8h, v28.8h, v0.h[1] //r[2]
    mul v29.8h, v29.8h, v0.h[1] //r[2]

    smlal   v9.4s, v24.4h, v0.h[0]
    smlal2 v10.4s, v24.8h, v0.h[0]
    smlal  v11.4s, v25.4h, v0.h[0]
    smlal2 v12.4s, v25.8h, v0.h[0]
    smlal  v13.4s, v26.4h, v0.h[0]
    smlal2 v14.4s, v26.8h, v0.h[0]
    smlal  v15.4s, v27.4h, v0.h[0]
    smlal2 v16.4s, v27.8h, v0.h[0]
    smlal  v17.4s, v28.4h, v0.h[0]
    smlal2 v18.4s, v28.8h, v0.h[0]
    smlal  v19.4s, v29.4h, v0.h[0]
    smlal2 v20.4s, v29.8h, v0.h[0]

    uzp2 v24.8h,  v9.8h, v10.8h
    uzp2 v25.8h, v11.8h, v12.8h
    uzp2 v26.8h, v13.8h, v14.8h
    uzp2 v27.8h, v15.8h, v16.8h
    uzp2 v28.8h, v17.8h, v18.8h
    uzp2 v29.8h, v19.8h, v20.8h

    smull   v9.4s, v3.4h, v3.4h //a0*a0
    smull2 v10.4s, v3.8h, v3.8h //a0*a0
    smull  v11.4s, v6.4h, v6.4h //a0*a0
    smull2 v12.4s, v6.8h, v6.8h //a0*a0

    smlal   v9.4s, v24.4h, v2.4h //-r[0]*zeta
    smlal2 v10.4s, v24.8h, v2.8h //-r[0]*zeta
    smlal  v11.4s, v25.4h, v1.4h //-r[0]*zeta
    smlal2 v12.4s, v25.8h, v1.8h //-r[0]*zeta

    smull  v13.4s, v26.4h, v1.4h //r[1]*zeta
    smull2 v14.4s, v26.8h, v1.8h //r[1]*zeta
    smull  v15.4s, v27.4h, v2.4h //r[1]*zeta
    smull2 v16.4s, v27.8h, v2.8h //r[1]*zeta

    smlal  v13.4s, v30.4h, v4.4h //-a0*a1
    smlal2 v14.4s, v30.8h, v4.8h //-a0*a1
    smlal  v15.4s, v31.4h, v7.4h //-a0*a1
    smlal2 v16.4s, v31.8h, v7.8h //-a0*a1

    uzp1 v24.8h,  v9.8h, v10.8h
    uzp1 v25.8h, v11.8h, v12.8h
    uzp1 v26.8h, v13.8h, v14.8h
    uzp1 v27.8h, v15.8h, v16.8h

    mul v24.8h, v24.8h, v0.h[1]
    mul v25.8h, v25.8h, v0.h[1]
    mul v26.8h, v26.8h, v0.h[1]
    mul v27.8h, v27.8h, v0.h[1]

    smlal   v9.4s, v24.4h, v0.h[0]
    smlal2 v10.4s, v24.8h, v0.h[0]
    smlal  v11.4s, v25.4h, v0.h[0]
    smlal2 v12.4s, v25.8h, v0.h[0]
    smlal  v13.4s, v26.4h, v0.h[0]
    smlal2 v14.4s, v26.8h, v0.h[0]
    smlal  v15.4s, v27.4h, v0.h[0]
    smlal2 v16.4s, v27.8h, v0.h[0]

    uzp2 v24.8h,  v9.8h, v10.8h //r[0]
    uzp2 v25.8h, v11.8h, v12.8h //r[0]
    uzp2 v26.8h, v13.8h, v14.8h //r[1]
    uzp2 v27.8h, v15.8h, v16.8h //r[1]

    smull   v9.4s, v28.4h, v4.4h //r[2]*a1
    smull2 v10.4s, v28.8h, v4.8h //r[2]*a1
    smull  v11.4s, v29.4h, v7.4h //r[2]*a1
    smull2 v12.4s, v29.8h, v7.8h //r[2]*a1

    smlal   v9.4s, v26.4h, v5.4h //r[1]*a2
    smlal2 v10.4s, v26.8h, v5.8h //r[1]*a2
    smlal  v11.4s, v27.4h, v8.4h //r[1]*a2
    smlal2 v12.4s, v27.8h, v8.8h //r[1]*a2

    uzp1 v30.8h,  v9.8h, v10.8h
    uzp1 v31.8h, v11.8h, v12.8h

    mul v30.8h, v30.8h, v0.h[1]
    mul v31.8h, v31.8h, v0.h[1]

    smlal   v9.4s, v30.4h, v0.h[0]
    smlal2 v10.4s, v30.8h, v0.h[0]
    smlal  v11.4s, v31.4h, v0.h[0]
    smlal2 v12.4s, v31.8h, v0.h[0]

    uzp2 v30.8h,  v9.8h, v10.8h
    uzp2 v31.8h, v11.8h, v12.8h

    smull   v9.4s, v30.4h, v1.4h //det*zeta
    smull2 v10.4s, v30.8h, v1.8h //det*zeta
    smull  v11.4s, v31.4h, v2.4h //det*zeta
    smull2 v12.4s, v31.8h, v2.8h //det*zeta

    smlal   v9.4s, v24.4h, v3.4h //r[0]*a[0]
    smlal2 v10.4s, v24.8h, v3.8h //r[0]*a[0]
    smlal  v11.4s, v25.4h, v6.4h //r[0]*a[0]
    smlal2 v12.4s, v25.8h, v6.8h //r[0]*a[0]

    uzp1 v13.8h,  v9.8h, v10.8h
    uzp1 v14.8h, v11.8h, v12.8h

    mul v13.8h, v13.8h, v0.h[1]
    mul v14.8h, v14.8h, v0.h[1]

    smlal   v9.4s, v13.4h, v0.h[0]
    smlal2 v10.4s, v13.8h, v0.h[0]
    smlal  v11.4s, v14.4h, v0.h[0]
    smlal2 v12.4s, v14.8h, v0.h[0]

    uzp2 v30.8h,  v9.8h, v10.8h
    uzp2 v31.8h, v11.8h, v12.8h

    st1 {v30.8h - v31.8h}, [stack_ptr], #32

    //Inverse test
    umin v30.8h, v30.8h, v31.8h
    uminv h30, v30.8h
    umov w15, v30.h[0]
    cmp w15, #0
    b.eq _loop_notinvertible

    st1 {v24.8h}, [dst], #16
    st1 {v26.8h}, [dst], #16
    st1 {v28.8h}, [dst], #16
    st1 {v25.8h}, [dst], #16
    st1 {v27.8h}, [dst], #16
    st1 {v29.8h}, [dst], #16

    subs counter2, counter2, #96
    b.ne _loop_inner1

    subs dst, dst, #288
    mov stack_ptr, sp
    mov counter2, #288

_loop_inner2:
    ld1 {v29.8h - v31.8h}, [stack_ptr], #48

    //Inverse
    //t1 = fqmul(a, a);    //10
    smull  v11.4s, v29.4h, v29.4h
    smull2 v12.4s, v29.8h, v29.8h
    smull  v13.4s, v30.4h, v30.4h
    smull2 v14.4s, v30.8h, v30.8h
    smull  v15.4s, v31.4h, v31.4h
    smull2 v16.4s, v31.8h, v31.8h

    uzp1 v17.8h, v11.8h, v12.8h
    uzp1 v18.8h, v13.8h, v14.8h
    uzp1 v19.8h, v15.8h, v16.8h

    mul v17.8h, v17.8h, v0.h[1]    
    mul v18.8h, v18.8h, v0.h[1]
    mul v19.8h, v19.8h, v0.h[1]
    
    smlal  v11.4s, v17.4h, v0.h[0]
    smlal2 v12.4s, v17.8h, v0.h[0]
    smlal  v13.4s, v18.4h, v0.h[0]
    smlal2 v14.4s, v18.8h, v0.h[0]
    smlal  v15.4s, v19.4h, v0.h[0]
    smlal2 v16.4s, v19.8h, v0.h[0]

    uzp2 v20.8h, v11.8h, v12.8h //t1
    uzp2 v21.8h, v13.8h, v14.8h //t1
    uzp2 v22.8h, v15.8h, v16.8h //t1

    //t2 = fqmul(t1, t1);  //100
    smull  v11.4s, v20.4h, v20.4h
    smull2 v12.4s, v20.8h, v20.8h
    smull  v13.4s, v21.4h, v21.4h
    smull2 v14.4s, v21.8h, v21.8h
    smull  v15.4s, v22.4h, v22.4h
    smull2 v16.4s, v22.8h, v22.8h

    uzp1 v17.8h, v11.8h, v12.8h
    uzp1 v18.8h, v13.8h, v14.8h
    uzp1 v19.8h, v15.8h, v16.8h

    mul v17.8h, v17.8h, v0.h[1]    
    mul v18.8h, v18.8h, v0.h[1]
    mul v19.8h, v19.8h, v0.h[1]
    
    smlal  v11.4s, v17.4h, v0.h[0]
    smlal2 v12.4s, v17.8h, v0.h[0]
    smlal  v13.4s, v18.4h, v0.h[0]
    smlal2 v14.4s, v18.8h, v0.h[0]
    smlal  v15.4s, v19.4h, v0.h[0]
    smlal2 v16.4s, v19.8h, v0.h[0]

    uzp2 v23.8h, v11.8h, v12.8h //t2
    uzp2 v24.8h, v13.8h, v14.8h //t2
    uzp2 v25.8h, v15.8h, v16.8h //t2

    //t2 = fqmul(t2, t2);  //1000
    smull  v11.4s, v23.4h, v23.4h
    smull2 v12.4s, v23.8h, v23.8h
    smull  v13.4s, v24.4h, v24.4h
    smull2 v14.4s, v24.8h, v24.8h
    smull  v15.4s, v25.4h, v25.4h
    smull2 v16.4s, v25.8h, v25.8h

    uzp1 v17.8h, v11.8h, v12.8h
    uzp1 v18.8h, v13.8h, v14.8h
    uzp1 v19.8h, v15.8h, v16.8h

    mul v17.8h, v17.8h, v0.h[1]    
    mul v18.8h, v18.8h, v0.h[1]
    mul v19.8h, v19.8h, v0.h[1]
    
    smlal  v11.4s, v17.4h, v0.h[0]
    smlal2 v12.4s, v17.8h, v0.h[0]
    smlal  v13.4s, v18.4h, v0.h[0]
    smlal2 v14.4s, v18.8h, v0.h[0]
    smlal  v15.4s, v19.4h, v0.h[0]
    smlal2 v16.4s, v19.8h, v0.h[0]

    uzp2 v23.8h, v11.8h, v12.8h //t2
    uzp2 v24.8h, v13.8h, v14.8h //t2
    uzp2 v25.8h, v15.8h, v16.8h //t2

    //t3 = fqmul(t2, t2);  //10000
    smull  v11.4s, v23.4h, v23.4h
    smull2 v12.4s, v23.8h, v23.8h
    smull  v13.4s, v24.4h, v24.4h
    smull2 v14.4s, v24.8h, v24.8h
    smull  v15.4s, v25.4h, v25.4h
    smull2 v16.4s, v25.8h, v25.8h

    uzp1 v17.8h, v11.8h, v12.8h
    uzp1 v18.8h, v13.8h, v14.8h
    uzp1 v19.8h, v15.8h, v16.8h

    mul v17.8h, v17.8h, v0.h[1]    
    mul v18.8h, v18.8h, v0.h[1]
    mul v19.8h, v19.8h, v0.h[1]
    
    smlal  v11.4s, v17.4h, v0.h[0]
    smlal2 v12.4s, v17.8h, v0.h[0]
    smlal  v13.4s, v18.4h, v0.h[0]
    smlal2 v14.4s, v18.8h, v0.h[0]
    smlal  v15.4s, v19.4h, v0.h[0]
    smlal2 v16.4s, v19.8h, v0.h[0]

    uzp2 v26.8h, v11.8h, v12.8h //t3
    uzp2 v27.8h, v13.8h, v14.8h //t3
    uzp2 v28.8h, v15.8h, v16.8h //t3

    //t1 = fqmul(t1, t2);  //1010
    smull  v11.4s, v20.4h, v23.4h
    smull2 v12.4s, v20.8h, v23.8h
    smull  v13.4s, v21.4h, v24.4h
    smull2 v14.4s, v21.8h, v24.8h
    smull  v15.4s, v22.4h, v25.4h
    smull2 v16.4s, v22.8h, v25.8h

    uzp1 v17.8h, v11.8h, v12.8h
    uzp1 v18.8h, v13.8h, v14.8h
    uzp1 v19.8h, v15.8h, v16.8h

    mul v17.8h, v17.8h, v0.h[1]    
    mul v18.8h, v18.8h, v0.h[1]
    mul v19.8h, v19.8h, v0.h[1]
    
    smlal  v11.4s, v17.4h, v0.h[0]
    smlal2 v12.4s, v17.8h, v0.h[0]
    smlal  v13.4s, v18.4h, v0.h[0]
    smlal2 v14.4s, v18.8h, v0.h[0]
    smlal  v15.4s, v19.4h, v0.h[0]
    smlal2 v16.4s, v19.8h, v0.h[0]

    uzp2 v20.8h, v11.8h, v12.8h //t1
    uzp2 v21.8h, v13.8h, v14.8h //t1
    uzp2 v22.8h, v15.8h, v16.8h //t1
    
    //t2 = fqmul(t1, t3);  //11010
    smull  v11.4s, v20.4h, v26.4h
    smull2 v12.4s, v20.8h, v26.8h
    smull  v13.4s, v21.4h, v27.4h
    smull2 v14.4s, v21.8h, v27.8h
    smull  v15.4s, v22.4h, v28.4h
    smull2 v16.4s, v22.8h, v28.8h

    uzp1 v17.8h, v11.8h, v12.8h
    uzp1 v18.8h, v13.8h, v14.8h
    uzp1 v19.8h, v15.8h, v16.8h

    mul v17.8h, v17.8h, v0.h[1]    
    mul v18.8h, v18.8h, v0.h[1]
    mul v19.8h, v19.8h, v0.h[1]
    
    smlal  v11.4s, v17.4h, v0.h[0]
    smlal2 v12.4s, v17.8h, v0.h[0]
    smlal  v13.4s, v18.4h, v0.h[0]
    smlal2 v14.4s, v18.8h, v0.h[0]
    smlal  v15.4s, v19.4h, v0.h[0]
    smlal2 v16.4s, v19.8h, v0.h[0]

    uzp2 v23.8h, v11.8h, v12.8h //t2
    uzp2 v24.8h, v13.8h, v14.8h //t2
    uzp2 v25.8h, v15.8h, v16.8h //t2

    //t2 = fqmul(t2, t2);  //110100
    smull  v11.4s, v23.4h, v23.4h
    smull2 v12.4s, v23.8h, v23.8h
    smull  v13.4s, v24.4h, v24.4h
    smull2 v14.4s, v24.8h, v24.8h
    smull  v15.4s, v25.4h, v25.4h
    smull2 v16.4s, v25.8h, v25.8h

    uzp1 v17.8h, v11.8h, v12.8h
    uzp1 v18.8h, v13.8h, v14.8h
    uzp1 v19.8h, v15.8h, v16.8h

    mul v17.8h, v17.8h, v0.h[1]    
    mul v18.8h, v18.8h, v0.h[1]
    mul v19.8h, v19.8h, v0.h[1]
    
    smlal  v11.4s, v17.4h, v0.h[0]
    smlal2 v12.4s, v17.8h, v0.h[0]
    smlal  v13.4s, v18.4h, v0.h[0]
    smlal2 v14.4s, v18.8h, v0.h[0]
    smlal  v15.4s, v19.4h, v0.h[0]
    smlal2 v16.4s, v19.8h, v0.h[0]

    uzp2 v23.8h, v11.8h, v12.8h //t2
    uzp2 v24.8h, v13.8h, v14.8h //t2
    uzp2 v25.8h, v15.8h, v16.8h //t2

    //t2 = fqmul(t2, a);   //110101
    smull  v11.4s, v23.4h, v29.4h
    smull2 v12.4s, v23.8h, v29.8h
    smull  v13.4s, v24.4h, v30.4h
    smull2 v14.4s, v24.8h, v30.8h
    smull  v15.4s, v25.4h, v31.4h
    smull2 v16.4s, v25.8h, v31.8h

    uzp1 v17.8h, v11.8h, v12.8h
    uzp1 v18.8h, v13.8h, v14.8h
    uzp1 v19.8h, v15.8h, v16.8h

    mul v17.8h, v17.8h, v0.h[1]    
    mul v18.8h, v18.8h, v0.h[1]
    mul v19.8h, v19.8h, v0.h[1]
    
    smlal  v11.4s, v17.4h, v0.h[0]
    smlal2 v12.4s, v17.8h, v0.h[0]
    smlal  v13.4s, v18.4h, v0.h[0]
    smlal2 v14.4s, v18.8h, v0.h[0]
    smlal  v15.4s, v19.4h, v0.h[0]
    smlal2 v16.4s, v19.8h, v0.h[0]

    uzp2 v23.8h, v11.8h, v12.8h //t2
    uzp2 v24.8h, v13.8h, v14.8h //t2
    uzp2 v25.8h, v15.8h, v16.8h //t2

    //t1 = fqmul(t1, t2);  //111111
    smull  v11.4s, v20.4h, v23.4h
    smull2 v12.4s, v20.8h, v23.8h
    smull  v13.4s, v21.4h, v24.4h
    smull2 v14.4s, v21.8h, v24.8h
    smull  v15.4s, v22.4h, v25.4h
    smull2 v16.4s, v22.8h, v25.8h

    uzp1 v17.8h, v11.8h, v12.8h
    uzp1 v18.8h, v13.8h, v14.8h
    uzp1 v19.8h, v15.8h, v16.8h

    mul v17.8h, v17.8h, v0.h[1]    
    mul v18.8h, v18.8h, v0.h[1]
    mul v19.8h, v19.8h, v0.h[1]
    
    smlal  v11.4s, v17.4h, v0.h[0]
    smlal2 v12.4s, v17.8h, v0.h[0]
    smlal  v13.4s, v18.4h, v0.h[0]
    smlal2 v14.4s, v18.8h, v0.h[0]
    smlal  v15.4s, v19.4h, v0.h[0]
    smlal2 v16.4s, v19.8h, v0.h[0]

    uzp2 v20.8h, v11.8h, v12.8h //t1
    uzp2 v21.8h, v13.8h, v14.8h //t1
    uzp2 v22.8h, v15.8h, v16.8h //t1

    //t2 = fqmul(t2, t2);  //1101010
    smull  v11.4s, v23.4h, v23.4h
    smull2 v12.4s, v23.8h, v23.8h
    smull  v13.4s, v24.4h, v24.4h
    smull2 v14.4s, v24.8h, v24.8h
    smull  v15.4s, v25.4h, v25.4h
    smull2 v16.4s, v25.8h, v25.8h

    uzp1 v17.8h, v11.8h, v12.8h
    uzp1 v18.8h, v13.8h, v14.8h
    uzp1 v19.8h, v15.8h, v16.8h

    mul v17.8h, v17.8h, v0.h[1]    
    mul v18.8h, v18.8h, v0.h[1]
    mul v19.8h, v19.8h, v0.h[1]
    
    smlal  v11.4s, v17.4h, v0.h[0]
    smlal2 v12.4s, v17.8h, v0.h[0]
    smlal  v13.4s, v18.4h, v0.h[0]
    smlal2 v14.4s, v18.8h, v0.h[0]
    smlal  v15.4s, v19.4h, v0.h[0]
    smlal2 v16.4s, v19.8h, v0.h[0]

    uzp2 v23.8h, v11.8h, v12.8h //t2
    uzp2 v24.8h, v13.8h, v14.8h //t2
    uzp2 v25.8h, v15.8h, v16.8h //t2

    //t2 = fqmul(t2, t2);  //11010100
    smull  v11.4s, v23.4h, v23.4h
    smull2 v12.4s, v23.8h, v23.8h
    smull  v13.4s, v24.4h, v24.4h
    smull2 v14.4s, v24.8h, v24.8h
    smull  v15.4s, v25.4h, v25.4h
    smull2 v16.4s, v25.8h, v25.8h

    uzp1 v17.8h, v11.8h, v12.8h
    uzp1 v18.8h, v13.8h, v14.8h
    uzp1 v19.8h, v15.8h, v16.8h

    mul v17.8h, v17.8h, v0.h[1]    
    mul v18.8h, v18.8h, v0.h[1]
    mul v19.8h, v19.8h, v0.h[1]
    
    smlal  v11.4s, v17.4h, v0.h[0]
    smlal2 v12.4s, v17.8h, v0.h[0]
    smlal  v13.4s, v18.4h, v0.h[0]
    smlal2 v14.4s, v18.8h, v0.h[0]
    smlal  v15.4s, v19.4h, v0.h[0]
    smlal2 v16.4s, v19.8h, v0.h[0]

    uzp2 v23.8h, v11.8h, v12.8h //t2
    uzp2 v24.8h, v13.8h, v14.8h //t2
    uzp2 v25.8h, v15.8h, v16.8h //t2

    //t2 = fqmul(t2, t2);  //110101000
    smull  v11.4s, v23.4h, v23.4h
    smull2 v12.4s, v23.8h, v23.8h
    smull  v13.4s, v24.4h, v24.4h
    smull2 v14.4s, v24.8h, v24.8h
    smull  v15.4s, v25.4h, v25.4h
    smull2 v16.4s, v25.8h, v25.8h

    uzp1 v17.8h, v11.8h, v12.8h
    uzp1 v18.8h, v13.8h, v14.8h
    uzp1 v19.8h, v15.8h, v16.8h

    mul v17.8h, v17.8h, v0.h[1]    
    mul v18.8h, v18.8h, v0.h[1]
    mul v19.8h, v19.8h, v0.h[1]
    
    smlal  v11.4s, v17.4h, v0.h[0]
    smlal2 v12.4s, v17.8h, v0.h[0]
    smlal  v13.4s, v18.4h, v0.h[0]
    smlal2 v14.4s, v18.8h, v0.h[0]
    smlal  v15.4s, v19.4h, v0.h[0]
    smlal2 v16.4s, v19.8h, v0.h[0]

    uzp2 v23.8h, v11.8h, v12.8h //t2
    uzp2 v24.8h, v13.8h, v14.8h //t2
    uzp2 v25.8h, v15.8h, v16.8h //t2

    //t2 = fqmul(t2, t2);  //1101010000
    smull  v11.4s, v23.4h, v23.4h
    smull2 v12.4s, v23.8h, v23.8h
    smull  v13.4s, v24.4h, v24.4h
    smull2 v14.4s, v24.8h, v24.8h
    smull  v15.4s, v25.4h, v25.4h
    smull2 v16.4s, v25.8h, v25.8h

    uzp1 v17.8h, v11.8h, v12.8h
    uzp1 v18.8h, v13.8h, v14.8h
    uzp1 v19.8h, v15.8h, v16.8h

    mul v17.8h, v17.8h, v0.h[1]    
    mul v18.8h, v18.8h, v0.h[1]
    mul v19.8h, v19.8h, v0.h[1]
    
    smlal  v11.4s, v17.4h, v0.h[0]
    smlal2 v12.4s, v17.8h, v0.h[0]
    smlal  v13.4s, v18.4h, v0.h[0]
    smlal2 v14.4s, v18.8h, v0.h[0]
    smlal  v15.4s, v19.4h, v0.h[0]
    smlal2 v16.4s, v19.8h, v0.h[0]

    uzp2 v23.8h, v11.8h, v12.8h //t2
    uzp2 v24.8h, v13.8h, v14.8h //t2
    uzp2 v25.8h, v15.8h, v16.8h //t2

    //t2 = fqmul(t2, t2);  //11010100000
    smull  v11.4s, v23.4h, v23.4h
    smull2 v12.4s, v23.8h, v23.8h
    smull  v13.4s, v24.4h, v24.4h
    smull2 v14.4s, v24.8h, v24.8h
    smull  v15.4s, v25.4h, v25.4h
    smull2 v16.4s, v25.8h, v25.8h

    uzp1 v17.8h, v11.8h, v12.8h
    uzp1 v18.8h, v13.8h, v14.8h
    uzp1 v19.8h, v15.8h, v16.8h

    mul v17.8h, v17.8h, v0.h[1]    
    mul v18.8h, v18.8h, v0.h[1]
    mul v19.8h, v19.8h, v0.h[1]
    
    smlal  v11.4s, v17.4h, v0.h[0]
    smlal2 v12.4s, v17.8h, v0.h[0]
    smlal  v13.4s, v18.4h, v0.h[0]
    smlal2 v14.4s, v18.8h, v0.h[0]
    smlal  v15.4s, v19.4h, v0.h[0]
    smlal2 v16.4s, v19.8h, v0.h[0]

    uzp2 v23.8h, v11.8h, v12.8h //t2
    uzp2 v24.8h, v13.8h, v14.8h //t2
    uzp2 v25.8h, v15.8h, v16.8h //t2

    //t2 = fqmul(t2, t2);  //110101000000
    smull  v11.4s, v23.4h, v23.4h
    smull2 v12.4s, v23.8h, v23.8h
    smull  v13.4s, v24.4h, v24.4h
    smull2 v14.4s, v24.8h, v24.8h
    smull  v15.4s, v25.4h, v25.4h
    smull2 v16.4s, v25.8h, v25.8h

    uzp1 v17.8h, v11.8h, v12.8h
    uzp1 v18.8h, v13.8h, v14.8h
    uzp1 v19.8h, v15.8h, v16.8h

    mul v17.8h, v17.8h, v0.h[1]    
    mul v18.8h, v18.8h, v0.h[1]
    mul v19.8h, v19.8h, v0.h[1]
    
    smlal  v11.4s, v17.4h, v0.h[0]
    smlal2 v12.4s, v17.8h, v0.h[0]
    smlal  v13.4s, v18.4h, v0.h[0]
    smlal2 v14.4s, v18.8h, v0.h[0]
    smlal  v15.4s, v19.4h, v0.h[0]
    smlal2 v16.4s, v19.8h, v0.h[0]

    uzp2 v23.8h, v11.8h, v12.8h //t2
    uzp2 v24.8h, v13.8h, v14.8h //t2
    uzp2 v25.8h, v15.8h, v16.8h //t2

    //t2 = fqmul(t1, t2);  //110101111111
    smull  v11.4s, v20.4h, v23.4h
    smull2 v12.4s, v20.8h, v23.8h
    smull  v13.4s, v21.4h, v24.4h
    smull2 v14.4s, v21.8h, v24.8h
    smull  v15.4s, v22.4h, v25.4h
    smull2 v16.4s, v22.8h, v25.8h

    uzp1 v17.8h, v11.8h, v12.8h
    uzp1 v18.8h, v13.8h, v14.8h
    uzp1 v19.8h, v15.8h, v16.8h

    mul v17.8h, v17.8h, v0.h[1]    
    mul v18.8h, v18.8h, v0.h[1]
    mul v19.8h, v19.8h, v0.h[1]
    
    smlal  v11.4s, v17.4h, v0.h[0]
    smlal2 v12.4s, v17.8h, v0.h[0]
    smlal  v13.4s, v18.4h, v0.h[0]
    smlal2 v14.4s, v18.8h, v0.h[0]
    smlal  v15.4s, v19.4h, v0.h[0]
    smlal2 v16.4s, v19.8h, v0.h[0]

    uzp2 v29.8h, v11.8h, v12.8h //det0
    uzp2 v30.8h, v13.8h, v14.8h //det1
    uzp2 v31.8h, v15.8h, v16.8h //det2

    neg v26.8h, v29.8h //-det0
    neg v27.8h, v30.8h //-det0
    neg v28.8h, v31.8h //-det0

    ld1 {v3.8h - v5.8h},  [dst]

    smull   v6.4s, v3.4h, v29.4h //r[0]*det0
    smull2  v7.4s, v3.8h, v29.8h //r[0]*det0
    smull   v8.4s, v4.4h, v29.4h //r[1]*det0
    smull2  v9.4s, v4.8h, v29.8h //r[1]*det0
    smull  v10.4s, v5.4h, v29.4h //r[2]*det0
    smull2 v11.4s, v5.8h, v29.8h //r[2]*det0

    uzp1 v12.8h,  v6.8h, v7.8h
    uzp1 v13.8h,  v8.8h, v9.8h
    uzp1 v14.8h, v10.8h, v11.8h

    mul v12.8h, v12.8h, v0.h[1]
    mul v13.8h, v13.8h, v0.h[1]
    mul v14.8h, v14.8h, v0.h[1]

    smlal   v6.4s, v12.4h, v0.h[0]
    smlal2  v7.4s, v12.8h, v0.h[0]
    smlal   v8.4s, v13.4h, v0.h[0]
    smlal2  v9.4s, v13.8h, v0.h[0]
    smlal  v10.4s, v14.4h, v0.h[0]
    smlal2 v11.4s, v14.8h, v0.h[0]

    uzp2 v3.8h,  v6.8h, v7.8h
    uzp2 v4.8h,  v8.8h, v9.8h
    uzp2 v5.8h, v10.8h, v11.8h

    st1 {v3.8h - v5.8h},  [dst], #48

     ld1 {v3.8h - v5.8h},  [dst]

    smull   v6.4s, v3.4h, v30.4h //r[0]*det0
    smull2  v7.4s, v3.8h, v30.8h //r[0]*det0
    smull   v8.4s, v4.4h, v30.4h //r[1]*det0
    smull2  v9.4s, v4.8h, v30.8h //r[1]*det0
    smull  v10.4s, v5.4h, v30.4h //r[2]*det0
    smull2 v11.4s, v5.8h, v30.8h //r[2]*det0

    uzp1 v12.8h,  v6.8h, v7.8h
    uzp1 v13.8h,  v8.8h, v9.8h
    uzp1 v14.8h, v10.8h, v11.8h

    mul v12.8h, v12.8h, v0.h[1]
    mul v13.8h, v13.8h, v0.h[1]
    mul v14.8h, v14.8h, v0.h[1]

    smlal   v6.4s, v12.4h, v0.h[0]
    smlal2  v7.4s, v12.8h, v0.h[0]
    smlal   v8.4s, v13.4h, v0.h[0]
    smlal2  v9.4s, v13.8h, v0.h[0]
    smlal  v10.4s, v14.4h, v0.h[0]
    smlal2 v11.4s, v14.8h, v0.h[0]

    uzp2 v3.8h,  v6.8h, v7.8h
    uzp2 v4.8h,  v8.8h, v9.8h
    uzp2 v5.8h, v10.8h, v11.8h

    st1 {v3.8h - v5.8h},  [dst], #48
    
    ld1 {v3.8h - v5.8h},  [dst]

    smull   v6.4s, v3.4h, v31.4h //r[0]*det0
    smull2  v7.4s, v3.8h, v31.8h //r[0]*det0
    smull   v8.4s, v4.4h, v31.4h //r[1]*det0
    smull2  v9.4s, v4.8h, v31.8h //r[1]*det0
    smull  v10.4s, v5.4h, v31.4h //r[2]*det0
    smull2 v11.4s, v5.8h, v31.8h //r[2]*det0

    uzp1 v12.8h,  v6.8h, v7.8h
    uzp1 v13.8h,  v8.8h, v9.8h
    uzp1 v14.8h, v10.8h, v11.8h

    mul v12.8h, v12.8h, v0.h[1]
    mul v13.8h, v13.8h, v0.h[1]
    mul v14.8h, v14.8h, v0.h[1]

    smlal   v6.4s, v12.4h, v0.h[0]
    smlal2  v7.4s, v12.8h, v0.h[0]
    smlal   v8.4s, v13.4h, v0.h[0]
    smlal2  v9.4s, v13.8h, v0.h[0]
    smlal  v10.4s, v14.4h, v0.h[0]
    smlal2 v11.4s, v14.8h, v0.h[0]

    uzp2 v3.8h,  v6.8h, v7.8h
    uzp2 v4.8h,  v8.8h, v9.8h
    uzp2 v5.8h, v10.8h, v11.8h

    st1 {v3.8h - v5.8h},  [dst], #48

    subs counter2, counter2, #144
    b.ne _loop_inner2


    subs counter1, counter1, #288
    b.ne _looptop_baseinv

    eor v0.16b, v0.16b, v0.16b

    st1 {v0.8h}, [sp], #16
    st1 {v0.8h}, [sp], #16
    st1 {v0.8h}, [sp], #16
    st1 {v0.8h}, [sp], #16
    st1 {v0.8h}, [sp], #16
    st1 {v0.8h}, [sp], #16

    eor x0, x0, x0

    ret

    .unreq    dst
    .unreq    src
    .unreq    zetas_ptr
    .unreq    counter1 
    .unreq    counter2

_loop_notinvertible:

    eor v0.16b, v0.16b, v0.16b
    
    st1 {v0.8h}, [sp], #16
    st1 {v0.8h}, [sp], #16
    st1 {v0.8h}, [sp], #16
    st1 {v0.8h}, [sp], #16
    st1 {v0.8h}, [sp], #16
    st1 {v0.8h}, [sp], #16

    mov x0, #1

    ret  


.align 4
zetas_mul:
    .hword 0x0d81, 0xcd7f, 0xff6d, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000
    .hword 0xfad5, 0x00a3, 0x0135, 0x03d5, 0xfdd3, 0xfefe, 0x00e8, 0xf970
    .hword 0x052b, 0xff5d, 0xfecb, 0xfc2b, 0x022d, 0x0102, 0xff18, 0x0690
    .hword 0xf987, 0xfb2f, 0x0090, 0x06a3, 0x0137, 0xfbdc, 0x0242, 0x0512
    .hword 0x0679, 0x04d1, 0xff70, 0xf95d, 0xfec9, 0x0424, 0xfdbe, 0xfaee
    .hword 0xfe6d, 0x0647, 0x0432, 0xff6c, 0x01bf, 0xf9e0, 0x0476, 0xfe6e
    .hword 0x0193, 0xf9b9, 0xfbce, 0x0094, 0xfe41, 0x0620, 0xfb8a, 0x0192
    .hword 0xfa7c, 0xfd91, 0x0357, 0x016d, 0xff9e, 0xff0c, 0x0197, 0x04c9
    .hword 0x0584, 0x026f, 0xfca9, 0xfe93, 0x0062, 0x00f4, 0xfe69, 0xfb37
    .hword 0x01a0, 0x02ab, 0xff97, 0x06b2, 0xfc05, 0x0425, 0x048b, 0x027e
    .hword 0xfe60, 0xfd55, 0x0069, 0xf94e, 0x03fb, 0xfbdb, 0xfb75, 0xfd82
    .hword 0x031e, 0x05d5, 0xfea1, 0x018c, 0xfde2, 0xfff7, 0x0650, 0xff75
    .hword 0xfce2, 0xfa2b, 0x015f, 0xfe74, 0x021e, 0x0009, 0xf9b0, 0x008b
    .hword 0xfc25, 0xfe1e, 0x0379, 0x00ee, 0xfa17, 0x01d2, 0xfbbf, 0xff9b
    .hword 0x03db, 0x01e2, 0xfc87, 0xff12, 0x05e9, 0xfe2e, 0x0441, 0x0065
    .hword 0x0351, 0xfe56, 0x0635, 0x05cf, 0x029f, 0x05b3, 0xfcf8, 0x00ff
    .hword 0xfcaf, 0x01aa, 0xf9cb, 0xfa31, 0xfd61, 0xfa4d, 0x0308, 0xff01
    .hword 0xfc0a, 0x0478, 0x01d8, 0xfb7f, 0xfebb, 0x05ef, 0xffe6, 0xfb9d
    .hword 0x03f6, 0xfb88, 0xfe28, 0x0481, 0x0145, 0xfa11, 0x001a, 0x0463
    .hword 0x0144, 0x04ce, 0x060b, 0xfdaf, 0xfe54, 0x04a8, 0x0430, 0xf9e4
    .hword 0xfebc, 0xfb32, 0xf9f5, 0x0251, 0x01ac, 0xfb58, 0xfbd0, 0x061c
    .hword 0x02b0, 0xfeb3, 0x03ff, 0xf96a, 0x0349, 0x0338, 0xffb9, 0x0633
    .hword 0xfd50, 0x014d, 0xfc01, 0x0696, 0xfcb7, 0xfcc8, 0x0047, 0xf9cd
    .hword 0x020a, 0xfebd, 0x047c, 0x0185, 0x04cf, 0x0180, 0x053f, 0x00a9
    .hword 0xfdf6, 0x0143, 0xfb84, 0xfe7b, 0xfb31, 0xfe80, 0xfac1, 0xff57
    .hword 0x0274, 0xfacf, 0xfbe0, 0xfc58, 0x0018, 0xfedb, 0x05f3, 0xfed4
    .hword 0xfd8c, 0x0531, 0x0420, 0x03a8, 0xffe8, 0x0125, 0xfa0d, 0x012c
    .hword 0xf98a, 0x037b, 0xfc3e, 0xffbd, 0x00b3, 0xfb67, 0x034c, 0xfe03
    .hword 0x0676, 0xfc85, 0x03c2, 0x0043, 0xff4d, 0x0499, 0xfcb4, 0x01fd
    .hword 0xf973, 0xf9e3, 0xfddb, 0xfa1c, 0x04a7, 0xfee8, 0xffd5, 0x029d
    .hword 0x068d, 0x061d, 0x0225, 0x05e4, 0xfb59, 0x0118, 0x002b, 0xfd63
    .hword 0xfd16, 0x02f1, 0x0302, 0xfbea, 0x06af, 0x059e, 0x02b2, 0x043b
    .hword 0x02ea, 0xfd0f, 0xfcfe, 0x0416, 0xf951, 0xfa62, 0xfd4e, 0xfbc5
    .hword 0x0426, 0x06bf, 0xfc8d, 0x0229, 0x0686, 0x0042, 0x0339, 0xff7b
    .hword 0xfbda, 0xf941, 0x0373, 0xfdd7, 0xf97a, 0xffbe, 0xfcc7, 0x0085
    .hword 0xf9ce, 0x027d, 0xfd58, 0xfc6b, 0x0284, 0xfe8c, 0xfb57, 0xfb90
    .hword 0x0632, 0xfd83, 0x02a8, 0x0395, 0xfd7c, 0x0174, 0x04a9, 0x0470
