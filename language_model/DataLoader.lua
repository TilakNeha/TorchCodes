local DataLoader = {}

DataLoader.vocab_mapping = {}

in_textfile = 'input.txt'
f = assert(io.open(in_textfile, "r"))
rawdata = f:read()

unordered = {}
repeat
    for char in rawdata:gmatch'.' do
        if not unordered[char] then unordered[char] = true end
    end
    rawdata = f:read()
until not rawdata
f:close()

ordered = {}
for char in pairs(unordered) do ordered[#ordered + 1] = char end
for i,char in ipairs(ordered) do DataLoader.vocab_mapping[char] = i end

f = assert(io.open(in_textfile, "r"))
data = {}
rawdata = f:read()
repeat
    for char in rawdata:gmatch'.' do
        data[#data + 1] = DataLoader.vocab_mapping[char]  
    end 
    rawdata = f:read()
until not rawdata

