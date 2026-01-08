use std::marker::PhantomData;
use std::ptr::NonNull;

use zfp_sys::{
    zfp_field, zfp_field_1d, zfp_field_2d, zfp_field_3d, zfp_field_4d, zfp_field_free, zfp_type,
};

use super::ZfpArray;

/// A `zfp` field.
#[derive(Debug)]
pub(super) struct ZfpField<'a> {
    field: NonNull<zfp_field>,
    phantom: PhantomData<&'a zfp_field>,
}

impl Drop for ZfpField<'_> {
    fn drop(&mut self) {
        unsafe {
            zfp_field_free(self.field.as_ptr());
        }
    }
}

impl<'a> ZfpField<'a> {
    pub(super) fn new(array: &'a mut ZfpArray, shape: &[usize]) -> Option<Self> {
        match shape.len() {
            1 => Self::new_1d(array, shape[0]),
            2 => Self::new_2d(array, shape[1], shape[0]),
            3 => Self::new_3d(array, shape[2], shape[1], shape[0]),
            4 => Self::new_4d(array, shape[3], shape[2], shape[1], shape[0]),
            _ => None,
        }
    }

    pub(super) unsafe fn new_empty(zfp_type_: zfp_type, shape: &[usize]) -> Option<Self> {
        let pointer = core::ptr::null_mut::<u8>().cast::<std::ffi::c_void>();
        match shape.len() {
            1 => NonNull::new(unsafe { zfp_field_1d(pointer, zfp_type_, shape[0]) }).map(|field| {
                Self {
                    field,
                    phantom: PhantomData,
                }
            }),
            2 => NonNull::new(unsafe { zfp_field_2d(pointer, zfp_type_, shape[1], shape[0]) }).map(
                |field| Self {
                    field,
                    phantom: PhantomData,
                },
            ),
            3 => NonNull::new(unsafe {
                zfp_field_3d(pointer, zfp_type_, shape[2], shape[1], shape[0])
            })
            .map(|field| Self {
                field,
                phantom: PhantomData,
            }),
            4 => NonNull::new(unsafe {
                zfp_field_4d(pointer, zfp_type_, shape[3], shape[2], shape[1], shape[0])
            })
            .map(|field| Self {
                field,
                phantom: PhantomData,
            }),
            _ => None,
        }
    }

    pub(super) fn new_1d(array: &mut ZfpArray, nx: usize) -> Option<Self> {
        if nx != array.len() {
            return None;
        }
        let field = unsafe { zfp_field_1d(array.as_mut_ptr(), array.zfp_type(), nx) };
        NonNull::new(field).map(|field| Self {
            field,
            phantom: PhantomData,
        })
    }

    pub(super) fn new_2d(array: &mut ZfpArray, nx: usize, ny: usize) -> Option<Self> {
        if nx * ny != array.len() {
            return None;
        }
        let field = unsafe { zfp_field_2d(array.as_mut_ptr(), array.zfp_type(), nx, ny) };
        NonNull::new(field).map(|field| Self {
            field,
            phantom: PhantomData,
        })
    }

    pub(super) fn new_3d(array: &'a mut ZfpArray, nx: usize, ny: usize, nz: usize) -> Option<Self> {
        if nx * ny * nz != array.len() {
            return None;
        }
        let field = unsafe { zfp_field_3d(array.as_mut_ptr(), array.zfp_type(), nx, ny, nz) };
        NonNull::new(field).map(|field| Self {
            field,
            phantom: PhantomData,
        })
    }

    pub(super) fn new_4d(
        array: &mut ZfpArray,
        nx: usize,
        ny: usize,
        nz: usize,
        nw: usize,
    ) -> Option<Self> {
        if nx * ny * nz * nw != array.len() {
            return None;
        }
        let field = unsafe { zfp_field_4d(array.as_mut_ptr(), array.zfp_type(), nx, ny, nz, nw) };
        NonNull::new(field).map(|field| Self {
            field,
            phantom: PhantomData,
        })
    }

    pub(super) const fn as_zfp_field(&self) -> *mut zfp_field {
        self.field.as_ptr()
    }
}
