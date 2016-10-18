% This make.m is for MATLAB and OCTAVE under Windows, Mac, and Unix
% Set up mex before running this

try
#    mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims read_data.cpp
#    mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims write_data.cpp
    oldflags = mkoctfile("-p","CFLAGS");
    newflags = [oldflags(1:(end-1)) " -std=c99"];
    setenv("CFLAGS", newflags);
    mex read_data.cpp
    mex write_data.cpp
    setenv("CFLAGS", oldflags);
catch
	fprintf('If make.m fails, please check README about detailed instructions.\n');
end
