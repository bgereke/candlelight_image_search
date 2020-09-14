<template>
  <div id="app">
    <div class="crop-search">
      <Cropper ref="cropper" class="image-to-crop" :src="image"/>
      <div class="button-wrapper">
        <span class="button" @click="$refs.file.click()">
          <input type="file" ref="file" @change="uploadImage($event)" accept="image/*">
          Upload
        </span>
        <span class="button" @click="cropSearch">Search</span>
      </div>
    </div>
    <div class="results-masonry">      
    </div>
  </div>
</template>

<script>
import {Cropper} from 'vue-advanced-cropper';

const background = chrome.extension.getBackgroundPage()

export default {
  name: "App",
  components: {
    Cropper
    },
  data() {
    return {
      append_cnt: 0,
      image:
        background.screenshotUrl,
      num_results: 20,
    };
  },
  methods: {
    cropSearch() {
      this.searched = true;
      const cropped = this.$refs.cropper.getResult();
			// if (cropped) {
			// 	const request_form = new FormData();
			// 	cropped.canvas.toBlob(blob => {
      //     request_form.append('query_image', blob);
      //     request_form.append('k', this.num_results);
      //     var start = Date.now();
      //     axios({
      //       method: 'post',
      //       url: 'http://localhost:8000/candlelight/',
      //       data: request_form,
      //       headers: {'Content-Type': 'multipart/form-data',
      //                 'Accept': 'image/jpeg' },
      //       responseType: 'arraybuffer'
      //     })
      //     .then((response) => {
      //       console.log(Date.now() - start)
      //       var responseBlob = new Blob([response.data], {type:"image/jpeg"});
      //       const newTab = window.open();
      //       newTab.document.body.innerHTML = `<h2>Query:</h2>
      //       <img src="${
      //         URL.createObjectURL(responseBlob)
      //         }" height="200px">
      //       <h2>Results:</h2>
      //           <img src="${
      //         URL.createObjectURL(responseBlob)
      //         }" height="200px">`
      //     }, (error) => {
      //       console.log(error);
      //     });
			// 	}, 'image/jpeg');
      // }      
      
      // var imagesLoaded = require('imagesloaded');
      var msnry = new Masonry( '.grid', {
        itemSelector: '.photo-item',
        columnWidth: '.grid__col-sizer',
        gutter: '.grid__gutter-sizer',
        percentPosition: true,
        stagger: 30,
        // nicer reveal transition
        visibleStyle: { transform: 'translateY(0)', opacity: 1 },
        hiddenStyle: { transform: 'translateY(100px)', opacity: 0 },
      });

      //------------------//

      // Get an API key for your demos at https://unsplash.com/developers
      var unsplashID = '9ad80b14098bcead9c7de952435e937cc3723ae61084ba8e729adb642daf0251';

      var infScroll = new InfiniteScroll( '.grid', {
        path: function() {
          return 'https://api.unsplash.com/photos?client_id='
            + unsplashID + '&page=' + this.pageIndex;
        },
        // load response as flat text
        responseType: 'text',
        outlayer: msnry,
        status: '.page-load-status',
        history: false,
      });

      // use element to turn HTML string into elements
      var proxyElem = document.createElement('div');

      infScroll.on( 'load', function( response ) {
        // parse response into JSON data
        var data = JSON.parse( response );
        // compile data into HTML
        var itemsHTML = data.map( getItemHTML ).join('');
        // convert HTML string into elements
        proxyElem.innerHTML = itemsHTML;
        // append item elements
        var items = proxyElem.querySelectorAll('.photo-item');
        imagesLoaded( items, function() {
          infScroll.appendItems( items );
          msnry.appended( items );
        });
        window.setTimeout(function() {
          msnry.layout();
          }, 1000);
      });

      // load initial page
      infScroll.loadNextPage();

      //------------------//

      var itemTemplateSrc = document.querySelector('#photo-item-template').innerHTML;

      function getItemHTML( photo ) {
        return microTemplate( itemTemplateSrc, photo );
      }

      // micro templating, sort-of
      function microTemplate( src, data ) {
        // replace {{tags}} in source
        return src.replace( /\{([\w\-_\.]+)\}/gi, function( match, key ) {
          // walk through objects to get value
          var value = data;
          key.split('.').forEach( function( part ) {
            value = value[ part ];
          });
          return value;
        });
      }
    },
    uploadImage(event) {
      // Reference to the DOM input element
      var input = event.target;
      // Ensure that you have a file before attempting to read it
      if (input.files && input.files[0]) {
        // create a new FileReader to read this image and convert to base64 format
        var reader = new FileReader();
        // Define a callback function to run, when FileReader finishes its job
        reader.onload = e => {
          // Note: arrow function used here, so that "this.imageData" refers to the imageData of Vue component
          // Read image as base64 and set to imageData
          this.image = e.target.result;
        };
        // Start the reader job - read file as a data url (base64 format)
        reader.readAsDataURL(input.files[0]);
      }
    },
  }
};

</script>

<style scoped>
.image-to-crop {
  border: solid 1px #EEE;
  max-height: 750px;
  width: 100%;
}

.button-wrapper {
  display: flex;
  justify-content: center;
  margin-top: 17px;
  margin-bottom: 17px;
}

.button {
  color: white;
  font-size: 16px;
  padding: 10px 20px;
  background: #3fb37f;
  cursor: pointer;
  transition: background 0.5s;
  font-family: Open Sans, Arial;
  margin: 0 10px;
}

.button:hover {
  background: #38d890;
}

.button input {
  display: none;
}

/* img {
  object-fit: cover;
  width: 100%;
  height: 100%;
  line-height: 0;
  display: block;
  border-radius: 25px 25px 25px 25px;
} */

body {
  font-family: sans-serif;
  line-height: 1.4;
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.grid__col-sizer,
.photo-item {
  width: 32%;
}

.grid__gutter-sizer {
  width: 2%;
}

.photo-item {
  margin-bottom: 10px;
  float: left;
}

.photo-item__image {
  display: block;
  max-width: 100%;
}

.photo-item__caption {
  position: absolute;
  left: 10px;
  bottom: 10px;
  margin: 0;
}

.photo-item__caption a {
  color: white;
  font-size: 0.8em;
  text-decoration: none;
}

.page-load-status {
  display: none; /* hidden by default */
  padding-top: 20px;
  border-top: 1px solid #DDD;
  text-align: center;
  color: #777;
}

/* loader ellips in separate pen CSS */


</style>