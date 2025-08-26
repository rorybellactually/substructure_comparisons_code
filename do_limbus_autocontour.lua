dofile('c:/software/runtime/lua/limbus_segmentation.lua')
require('lfs')

-- may or may not need to scandir
-- we have all files as DICOM somewhere: i think it will be easiest to work with them in that form rather than try and have to deal with nifty-conversion in lua

-- wm.Scan[2]=wm.Scan[1]:burn(wm.Delineation.Atrium_L)+wm.Scan[1]:burn(wm.Delineation.Atrium_R)
-- wm.Scan[2]=wm.Scan[1]:burn(wm.Delineation.Atrium_L, 255, true)



--requite_input_folder = [[\\130.88.233.166\data\Msc_Minghao\REQUITE]]
--lymphoma_input_folder = [[\\130.88.233.166\data\Msc_Minghao\Lymphoma]]
--paediatrics_input_folder = [[\\130.88.233.166\data\Msc_Minghao\Paediatrics_withCardiacSubstructs]]

--wm.Scan[3].Data:max()
--"Double  Value: 0  [-1E100, 1E100]"
--wm.Scan[3].Data:max()
--"Double  Value: 9  [-1E100, 1E100]"
--wm.Scan[3].Data:max()
-- wm.Scan[2]=wm.Scan[1]:burn(wm.Delineation.Atrium_L)
--wm.Scan[1]:write_nifty(


function scandir(folder)
  local r = {}
  for f in lfs.dir(folder) do
    if f:sub(1, 1)~='.' then
      table.insert(r, f)
    end
  end
  return r
end

function folderexists(path)
  if (lfs.attributes(path, "mode") == "directory") then
    return true
  end
  return false
end

function fileexists(path)
  local f = io.open(path, "r")
  if f then f:close() return true end
  return false
end

function findctslice(folder)
  local r = scandir(folder)
  for i=1, #r do
    if string.match(r[i], 'CT_.*dcm') then
      return r[i], folder
    end
    if folderexists(folder .. '\\' .. r[i]) then
      return findctslice(folder .. '\\' .. r[i])  
    end
  end
end


names = {'lymphoma', 'requite', 'paeds'}
--for _, name in ipairs(names) do
--    if name == 'lymphoma' then
--      f = scandir(lymphoma_input_folder)
--      slice, folder = findctslice(inputfolder..'\\'..v..'\\'..w..'\\'..'AVG AVG')
--    elseif name == 'requite' then
--      f = scandir(requite_input_folder)
--    else
--      f = scandir(paediatrics_input_folder)
--    end
    
--end 

function find_file_in_list(filelist, pattern, filepath)
  for _, filename in ipairs(filelist) do
      if string.match(filename, pattern) then
        return (filepath .. "\\" .. filename)
      end
  end
  return nil
end

function store_patient_numbers_in_table(filepath)
  local lines = {}
  local file = io.open(filepath, "r")
  if not file then
    print("Failed to open file: " .. filepath)
    return nil
  end
  for line in file:lines() do
    table.insert(lines, line)
  end
  file:close()
  return lines
end






lymphoma_input_folder = [[\\130.88.233.166\data\Msc_Minghao\Lymphoma]]
lymphoma_patient_numbers = store_patient_numbers_in_table([[\\130.88.233.166\data\Rory\CohortStudy\patient_nos.txt]])
for i, patientID in ipairs(lymphoma_patient_numbers) do
 
  local patient_path = lymphoma_input_folder .. "\\" .. patientID
  if folderexists(patient_path) then
    
    print("patient path folder found")
   
    local f = scandir(patient_path)
    local niftyFullFilepath = find_file_in_list(f, "ct%.nii%.gz", patient_path)

    if niftyFullFilepath then
      
      wm.Scan[1]:load(niftyFullFilepath)
      print("loaded nifty file")
      wm.Scan[1] = wm.Scan[1] + 1024 -- convert HU to CT values
    
      --if not fileexists(lymphoma_input_folder .. '\\' .. patientID .. "\\" ..'RTSTRUCT_Limb_'..wm.Scan[1].Properties.SeriesInstanceUID..'.dcm') then
      if not fileexists(lymphoma_input_folder .. '\\' .. patientID .. "\\" ..'RTSTRUCT_Limb_'..'.dcm') then

            --print(lymphoma_input_folder .. '\\' .. patientID .. "\\" ..'RTSTRUCT_Limb_'..wm.Scan[1].Properties.SeriesInstanceUID..'.dcm')
            print(lymphoma_input_folder .. '\\' .. patientID .. "\\" ..'RTSTRUCT_Limb_'..'.dcm')

            limbus_accolade()
            
            local contour_dir = lymphoma_input_folder .. '\\' .. patientID .. "\\" .. "LimbusContour" 
            local output_file = contour_dir .. "\\" .. 'RTSTRUCT_Limb_'..wm.Scan[1].Properties.SeriesInstanceUID..'.dcm'
            
            if not folderexists(contour_dir) then
              os.execute('mkdir "' .. contour_dir .. '"')
            end
            
            wm.Delineation:export(wm.Scan[1], output_file)
          
      end
  end
    
  else
    print("path does not exist: " .. patient_path)
  end

end