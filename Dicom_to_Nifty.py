import os
import sys
import logging
import SimpleITK as sitk

# Setup logging
log_file = r"/home/donal/data/server2/Msc_Minghao/PaediatricsLog/autoseg_processing_log.txt"
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Add the path to the RaystationUtils.py file to the system path
utils_path = r'/home/donal/data/server2/Msc_Minghao/RaystationUtils.py'
utils_dir = os.path.dirname(utils_path)

if os.path.exists(utils_dir) and utils_dir not in sys.path:
    sys.path.append(utils_dir)
import RaystationUtils as utils

# 设置路径
ctScansPath = r'/home/donal/data/server2/Msc_Minghao/Paediatrics_withCardiacSubstructs'

def dicom_to_nifti_converter(dicom_path, patient_id=None):
    """
    将DICOM文件转换为NIfTI格式，输出到原目录位置
    
    Args:
        dicom_path (str): DICOM文件夹路径
        patient_id (str, optional): 患者ID，如果未提供则使用文件夹名称
    
    Returns:
        str: NIfTI文件路径，如果失败返回None
    """
    try:
        # 如果没有提供patient_id，使用文件夹名称
        if patient_id is None:
            patient_id = os.path.basename(os.path.normpath(dicom_path))
        
        # NIfTI文件路径（输出到原目录）
        nifti_file_path = os.path.join(dicom_path, f"ct.nii.gz")
        
        # 检查文件是否已存在
        if os.path.exists(nifti_file_path):
            logging.info(f"NIfTI file already exists: {nifti_file_path}")
            print(f"NIfTI file already exists: {nifti_file_path}")
            return nifti_file_path
        
        # 读取DICOM序列
        logging.info(f"Reading DICOM series from: {dicom_path}")
        print(f"Reading DICOM series from: {dicom_path}")
        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(dicom_path)
        
        if not series_IDs:
            logging.error(f"No DICOM series found in: {dicom_path}")
            print(f"No DICOM series found in: {dicom_path}")
            return None
        
        # 选择切片数最多的序列
        best_series_id = ""
        max_slices = 0
        logging.info(f"Found {len(series_IDs)} series. Selecting the one with most slices.")
        print(f"Found {len(series_IDs)} series. Selecting the one with most slices.")
        
        for series_id in series_IDs:
            file_names = reader.GetGDCMSeriesFileNames(dicom_path, series_id)
            num_slices = len(file_names)
            logging.debug(f"Series ID: {series_id} has {num_slices} slices.")
            print(f"Series ID: {series_id} has {num_slices} slices.")
            if num_slices > max_slices:
                max_slices = num_slices
                best_series_id = series_id
        
        if max_slices < 10:
            logging.warning(f"Warning: Series has only {max_slices} slices, this may cause issues.")
            print(f"Warning: Series has only {max_slices} slices, this may cause issues.")
        
        logging.info(f"Selected series ID: {best_series_id} with {max_slices} slices")
        print(f"Selected series ID: {best_series_id} with {max_slices} slices")
        
        # 读取选定的序列
        file_names = reader.GetGDCMSeriesFileNames(dicom_path, best_series_id)
        reader.SetFileNames(file_names)
        image = reader.Execute()
        
        # 转换为float32以获得更好的处理效果
        image = sitk.Cast(image, sitk.sitkFloat32)
        
        # 保存为NIfTI格式
        sitk.WriteImage(image, nifti_file_path)
        logging.info(f"Successfully converted DICOM to NIfTI: {nifti_file_path}")
        print(f"Successfully converted DICOM to NIfTI: {nifti_file_path}")
        
        return nifti_file_path
        
    except Exception as e:
        logging.error(f"Failed to convert DICOM to NIfTI for {patient_id}: {e}")
        print(f"Failed to convert DICOM to NIfTI for {patient_id}: {e}")
        return None

def batch_convert_dicom_to_nifti(root_path):
    """
    批量转换文件夹中的DICOM文件到NIfTI格式
    
    Args:
        root_path (str): 包含多个患者DICOM文件夹的根路径
    """
    if not os.path.exists(root_path):
        print(f"Root path does not exist: {root_path}")
        return
    
    success_count = 0
    fail_count = 0
    
    # 遍历根目录下的所有子文件夹
    for patient_folder in os.listdir(root_path):
        patient_path = os.path.join(root_path, patient_folder)
        
        # 跳过非文件夹
        if not os.path.isdir(patient_path):
            continue
            
        print(f"\nProcessing patient: {patient_folder}")
        result = dicom_to_nifti_converter(patient_path, patient_folder)
        
        if result:
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n=== Conversion Summary ===")
    print(f"Successfully converted: {success_count}")
    print(f"Failed conversions: {fail_count}")
    print(f"Total processed: {success_count + fail_count}")

# 使用示例
if __name__ == "__main__":
    try:
        # 从文件读取患者ID列表
        patient_ids = utils.txt_to_numpy_array(r'/home/donal/data/server2/Msc_Minghao/PaediatricsLog/Patients.txt')
        
        success_count = 0
        fail_count = 0
        
        logging.info(f"Starting batch conversion for {len(patient_ids)} patients")
        print(f"Starting batch conversion for {len(patient_ids)} patients")
        
        for patient_id in patient_ids:
            patient_id = patient_id.strip()  # 去除可能的空格
            if not patient_id:  # 跳过空行
                continue
                
            patient_dicom_path = os.path.join(ctScansPath, patient_id)
            
            if not os.path.exists(patient_dicom_path):
                logging.warning(f"Patient folder does not exist: {patient_dicom_path}")
                print(f"Patient folder does not exist: {patient_dicom_path}")
                fail_count += 1
                continue
            
            logging.info(f"Processing patient: {patient_id}")
            print(f"\nProcessing patient: {patient_id}")
            
            result = dicom_to_nifti_converter(patient_dicom_path, patient_id)
            
            if result:
                success_count += 1
            else:
                fail_count += 1
        
        # 记录总结
        logging.info(f"=== Conversion Summary ===")
        logging.info(f"Successfully converted: {success_count}")
        logging.info(f"Failed conversions: {fail_count}")
        logging.info(f"Total processed: {success_count + fail_count}")
        
        print(f"\n=== Conversion Summary ===")
        print(f"Successfully converted: {success_count}")
        print(f"Failed conversions: {fail_count}")
        print(f"Total processed: {success_count + fail_count}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        print(f"Error in main execution: {e}")
        
        # 如果患者列表文件不存在，提供备用方案
        print("\nFallback: Processing all folders in ctScansPath...")
        batch_convert_dicom_to_nifti(ctScansPath)
