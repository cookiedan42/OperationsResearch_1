try:
    from ipython import display
except:
    pass
def subscript(i):
    # ima let you toggle between subscript and full height numbers
    # return i
    out = ""
    for j in str(i):
        out +=chr(ord(u'\u2080')+ int(j))
    return out
# LP Object
# duality finding method
def displayHelper(item):
    #display works on jupyter but might not always be included?
    try:
        display(item)
    except:
        print(item)