import cairosvg
#import rsvg

'''img = cairo.ImageSurface(cairo.FORMAT_ARGB32, 640,480)

ctx = cairo.Context(img)

## handle = rsvg.Handle(<svg filename>)
# or, for in memory SVG data:
handle= rsvg.Handle(None, str(<svg data>))

handle.render_cairo(ctx)

img.write_to_png("svg.png")'''


from cairosvg import svg2png

svg_code = """
    <svg

   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   viewBox="0 0 533.33331 533.33331"
   height="533.33331"
   width="533.33331"
   xml:space="preserve"
   id="svg2"
   version="1.1"><metadata
     id="metadata8"><rdf:RDF><cc:Work
         rdf:about=""><dc:format>image/svg+xml</dc:format><dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" /></cc:Work></rdf:RDF></metadata><defs
     id="defs6"><clipPath
       id="clipPath18"
       clipPathUnits="userSpaceOnUse"><path
         id="path16"
         d="M 0,400 H 400 V 0 H 0 Z" /></clipPath></defs><g
     transform="matrix(1.3333333,0,0,-1.3333333,0,533.33333)"
     id="g10"><g
       id="g12"><g
         clip-path="url(#clipPath18)"
         id="g14"><g
           transform="translate(126.0366,200.3213)"
           id="g20"><path
             id="path22"
             style="fill:none;stroke:#000000;stroke-width:20;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             d="m 0,0 h 149.342 c 0.275,0 0.6,-0.202 0.721,-0.449 l 73.68,-149.424 c 0.121,-0.246 0.121,-0.246 0,0 L 76.258,149.23 c -0.122,0.247 -0.323,0.248 -0.447,0.003 L -76.037,-150.321" /></g><g
           transform="translate(220,130)"
           id="g24"></g></g></g></g></svg>
"""

path="/Users/kashefkarim/desktop/BA/Price/"
name='monogramm'
for i in range(1,1500):
    f = open(path+name+str(i)+".svg", "r")
    image=f.read()
    svg2png(bytestring=image,write_to='BA/Exp/monogram'+str(i-5)+'.png')


svg2png(bytestring=svg_code,write_to='BA/train/train_a1.png')


