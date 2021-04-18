import os
import re

# Container of attributes. Attributes are dynamically created.


class DATAttributes(object):
    pass

# DAT file parser. Does not matter the attributes. This class just parses file and reads all the attributes.
# A separate validator class should be used to check that attributes are valid.


class DATParser(object):
    @staticmethod
    def _tryParse(x):
        # try parsing x as integer
        try:
            return(int(x))
        except ValueError:
            pass

        # try parsing x as float
        try:
            return(float(x))
        except ValueError:
            pass

        # try parsing x as bool
        if(x in ['True',  'true',  'TRUE', 'T', 't']):
            return(True)
        if(x in ['False', 'false', 'FALSE', 'F', 'f']):
            return(False)

        # x cannot be parsed, leave it as is
        return(x)

    @staticmethod
    def _openFile(filePath):
        if(not os.path.exists(filePath)):
            raise Exception('The file (%s) does not exist' % filePath)
        return(open(filePath, 'r'))

    @staticmethod
    def parse(filePath):
        fileHandler = DATParser._openFile(filePath)
        fileContent = fileHandler.read()
        fileHandler.close()

        datAttr = DATAttributes()

        # lines not starting with <spaces>[a-zA-Z] are ignored.
        # comments can be added using for instance '$','//','#', ...

        # parse scalar attributes
        pattern = re.compile(r'^[\s]*([a-zA-Z][\w]*)[\s]*\=[\s]*([\w\/\.\-]+)[\s]*\;', re.M)
        entries = pattern.findall(fileContent)
        for entry in entries:
            datAttr.__dict__[entry[0]] = DATParser._tryParse(entry[1])

        # parse 1-dimension vector attributes
        pattern = re.compile(
            r'^[\s]*([a-zA-Z][\w]*)[\s]*\=[\s]*\[[\s]*(([\w\/\.\-]+[\s]*)+)\][\s]*\;', re.M)
        entries = pattern.findall(fileContent)
        for entry in entries:
            pattern2 = re.compile(r'([\w\/\.]+)[\s]*')
            values = pattern2.findall(entry[1])
            datAttr.__dict__[entry[0]] = map(DATParser._tryParse, values)

        # parse 2-dimension vector attributes
        pattern = re.compile(
            r'^[\s]*([a-zA-Z][\w]*)[\s]*\=[\s]*\[(([\s]*\[[\s]*(([\w\/\.\-]+[\s]*)+)\][\s]*)+)[\s]*\][\s]*\;', re.M)
        entries = pattern.findall(fileContent)
        for entry in entries:
            pattern2 = re.compile(r'[\s]*\[[\s]*(([\w\/\.\-]+[\s]*)+)\][\s]*')
            entries2 = pattern2.findall(entry[1])
            values = []
            for entry2 in entries2:
                pattern2 = re.compile(r'([\w\/\.\-]+)[\s]*')
                values2 = pattern2.findall(entry2[0])
                values.append(map(DATParser._tryParse, values2))
            datAttr.__dict__[entry[0]] = values

        return(datAttr)
