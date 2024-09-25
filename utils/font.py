# %%
import matplotlib.font_manager as fm
from matplotlib import rcParams
import matplotlib.pyplot as plt

def font_init(font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'):
    """
    Set up a Chinese font for matplotlib.
    
    Args:
    font_path (str): Path to the Chinese font file. Defaults to Noto Sans CJK.
    
    Returns:
    FontProperties: A FontProperties object for the Chinese font.
    """
    # Add the font file
    fm.fontManager.addfont(font_path)
    
    # Create a FontProperties object
    font_prop = fm.FontProperties(fname=font_path)
    
    # Set up the font family
    plt.rcParams['font.family'] = font_prop.get_name()

    # If it's a Unicode font, you might also want to set the following:
    plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
    
    return font_prop

cjk = font_init()
noto_sans = font_init('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
dejavu_mono = font_init('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf')
# %%
