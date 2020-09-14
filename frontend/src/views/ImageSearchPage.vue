<template>
  <div class="page">
    <div class="crop-search">
      <Cropper ref="cropper" class="image-to-crop" :src="image"/>
      <div class="button-wrapper">
        <span class="button" @click="$refs.file.click()">
          <input type="file" ref="file" @change="uploadImage($event)" accept="image/*">
          Upload
        </span>
        <span class="button" @click="onSubmit">Search</span>
      </div>
    </div>
    
    <br />

    <div
      v-infinite-scroll="searchAllImages"
      infinite-scroll-disabled="busy"
      infinite-scroll-distance="10"
    >
      <div
        v-masonry="containerId"
        transition-duration="0.3s"
        item-selector=".item"
        gutter="5"
        fit-width="true"
        class="masonry-container"
      >
        <div>
          <img
            :src="item.previewURL"
            v-masonry-tile
            class="item"
            v-for="(item, index) in images"
            :key="index"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { requestsMixin } from "../mixins/requestsMixin";
import {Cropper} from 'vue-advanced-cropper';
const background = chrome.extension.getBackgroundPage()

export default {
  mixins: [requestsMixin],
  components: {
    Cropper
    },
  data() {
    return {
      form: {},
      page: 1,
      containerId: null,
      images: [],
      append_cnt: 0,
      image:
        background.screenshotUrl,
      num_results: 20
    };
  },
  methods: {
    async onSubmit() {
    //   const isValid = await this.$refs.observer.validate();
    //   if (!isValid) {
    //     return;
    //   }
      this.page = 1;
      this.form.keyword = 'dog';
      await this.searchAllImages();
    },

    async searchAllImages() {
      if (!this.form.keyword) {
        return;
      }
      const response = await this.searchImages(this.form.keyword, this.page);
      if (this.page == 1) {
        this.images = response.data.hits;
      } else {
        this.images = this.images.concat(response.data.hits);
      }
      this.page++;
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
    }
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

img {
  /* object-fit: cover;
  width: 100%;
  height: 100%;
  line-height: 0; */
  display: block;
  border-radius: 25px 25px 25px 25px;
}


</style>