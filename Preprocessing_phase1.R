library(dismo)
library(dplyr)
library(ggplot2)
library(openxlsx)
library(readxl)
library(sp)
library(sf)
library(tmap)
library( spThin )
library(mopa)
tmap::tmap_mode("plot")
par(mar=c(1,1,1,1))
library(rnaturalearth)

# read in a raster mask 
r1 <- raster("wc2.1_2.5m_bio/wc2.1_2.5m_bio_1.tif")
sp1 <- st_read("MAR_shape/ne_10m_admin_0_countries.shp")
mar <- sp1 %>%
  dplyr::filter(ADMIN == "Morocco")
ex1 <- raster::extent(mar)
r2 <- raster::crop(x = r1, y = ex1)
mask <- raster::mask(x = r2, mask = mar)
qtm(mask)

## Read the birds data
data_moussieri <- read_excel("redstarts_birds.xlsx", sheet = "Moussieri")
data_ochruros <- read_excel("redstarts_birds.xlsx", sheet = "Ochruros")
data_phoenicurus <- read_excel("redstarts_birds.xlsx", sheet = "Phoenicurus")


# this function filter the occurence data (thinning) and generate pseudo-absence data
thin_ganerate_pseudo_abs <- function(species_data, mask, samples_size, thin_dist, pseu_buffer) {
  thinned_species <- thin( loc.data = species_data, lat.col = "Latitude", long.col = "Longitude", 
                       spec.col = "species", thin.par = thin_dist, reps = 10, locs.thinned.list.return = TRUE, 
                       write.files = FALSE, write.log.file = FALSE)
  thinned_species= thinned_species[[1]]
  thinned_species["presence"] <- 1
  #convert dataframe to spatilpoints
  dataSelection= select(thinned_species, Longitude, Latitude)
  coordinates(dataSelection) <- c('Longitude', 'Latitude')
  projection(dataSelection) <- CRS('+proj=longlat +datum=WGS84')
  # draw a buffer around each presence point 
  x <- circles(dataSelection, d=pseu_buffer, lonlat=TRUE)
  pol <- polygons(x)
  # sampling
  samp1 <- spsample(pol, samples_size, type='random', iter=25)
  # get unique cells
  cells <- cellFromXY(mask, samp1)
  cells <- unique(cells)
  length(cells)
  xy <- xyFromCell(mask, cells)
  
  last_samples_des <- as.data.frame(xy)
  last_samples_des= select(last_samples_des, x, y)
  last_samples_des <- na.omit(last_samples_des) 
  coordinates(last_samples_des) <- c('x', 'y')
  projection(last_samples_des) <- CRS('+proj=longlat +datum=WGS84')
  
  #exclude generated pseudo-absence data that are outside the morocco study area
  polyy <- ne_countries(country = 'morocco')
  projection(polyy) <- CRS('+proj=longlat +datum=WGS84')
  last_samples_des<-last_samples_des[!is.na(over(last_samples_des, polyy[1])),]
  last_samples_des_df<- as.data.frame(last_samples_des)
  colnames(last_samples_des_df) <- c('Longitude', 'Latitude')
  last_samples_des_df["presence"] <- 0
  
  # filtering (thinning the generated pseudo-absence data)
  thinned_ps <- thin( loc.data = last_samples_des_df, lat.col = "Latitude", long.col = "Longitude", 
                           spec.col = "presence", thin.par = 2, reps = 1, locs.thinned.list.return = TRUE, 
                           write.files = FALSE, write.log.file = FALSE)
  thinned_ps= thinned_ps[[1]]
  thinned_ps["presence"] <- 0
  final_data <- rbind(thinned_species, thinned_ps)
  return (final_data)
}

# calling the function: thinning and generating pseudo-absence data
final_data_moussieri <- thin_ganerate_pseudo_abs(data_moussieri, mask, 3700, 2, 250000)
final_data_ochruros <- thin_ganerate_pseudo_abs(data_ochruros, mask, 4500, 2, 250000)
final_data_phoenicurus <- thin_ganerate_pseudo_abs(data_phoenicurus, mask, 4900, 2, 250000)

# select the coordinates
data_moussieri_sel= select(final_data_moussieri, Longitude, Latitude)
data_ochruros_sel= select(final_data_ochruros, Longitude, Latitude)
data_phoenicurus_sel= select(final_data_phoenicurus, Longitude, Latitude)


# read the environmental data 
bio_1= stack("wc2.1_2.5m_bio/wc2.1_2.5m_bio_1.tif")
bio_2= stack("wc2.1_2.5m_bio/wc2.1_2.5m_bio_2.tif")
bio_3= stack("wc2.1_2.5m_bio/wc2.1_2.5m_bio_3.tif")
bio_4= stack("wc2.1_2.5m_bio/wc2.1_2.5m_bio_4.tif")
bio_5= stack("wc2.1_2.5m_bio/wc2.1_2.5m_bio_5.tif")
bio_6= stack("wc2.1_2.5m_bio/wc2.1_2.5m_bio_6.tif")
bio_7= stack("wc2.1_2.5m_bio/wc2.1_2.5m_bio_7.tif")
bio_8= stack("wc2.1_2.5m_bio/wc2.1_2.5m_bio_8.tif")
bio_9= stack("wc2.1_2.5m_bio/wc2.1_2.5m_bio_9.tif")
bio_10= stack("wc2.1_2.5m_bio/wc2.1_2.5m_bio_10.tif")
bio_11= stack("wc2.1_2.5m_bio/wc2.1_2.5m_bio_11.tif")
bio_12= stack("wc2.1_2.5m_bio/wc2.1_2.5m_bio_12.tif")
bio_13= stack("wc2.1_2.5m_bio/wc2.1_2.5m_bio_13.tif")
bio_14= stack("wc2.1_2.5m_bio/wc2.1_2.5m_bio_14.tif")
bio_15= stack("wc2.1_2.5m_bio/wc2.1_2.5m_bio_15.tif")
bio_16= stack("wc2.1_2.5m_bio/wc2.1_2.5m_bio_16.tif")
bio_17= stack("wc2.1_2.5m_bio/wc2.1_2.5m_bio_17.tif")
bio_18= stack("wc2.1_2.5m_bio/wc2.1_2.5m_bio_18.tif")
bio_19= stack("wc2.1_2.5m_bio/wc2.1_2.5m_bio_19.tif")
elevation= stack("wc2.1_2.5m_elev/wc2.1_2.5m_elev.tif")
LC= stack("wc2.1_2.5m_bio/MCD12Q1_LC4_2001_001.tif")
LC <- projectRaster(LC, crs = "+proj=longlat +datum=WGS84")

# rename the variables
names (bio_1) <- "bio1";names (bio_2) <- "bio2";names (bio_3) <- "bio3";names (bio_4) <- "bio4";names (bio_5) <- "bio5";names (bio_6) <- "bio6";names (bio_7) <- "bio7";names (bio_8) <- "bio8";names (bio_9) <- "bio9";names (bio_10) <- "bio10";
names (bio_11) <- "bio11";names (bio_12) <- "bio12";names (bio_13) <- "bio13";names (bio_14) <- "bio14";names (bio_15) <- "bio15";names (bio_16) <- "bio16";names (bio_17) <- "bio17";names (bio_18) <- "bio18";names (bio_19) <- "bio19";names (elevation) <- "elv";
names (LC) <- "LC"


# this function maps the environmental data with birds locations (presence-absence data) 
map_envs_toXY <- function(species_data, species_data_sel) {
  bio_1_ext= extract(bio_1, species_data_sel)
  bio_2_ext= extract(bio_2, species_data_sel)
  bio_3_ext= extract(bio_3, species_data_sel)
  bio_4_ext= extract(bio_4, species_data_sel)
  bio_5_ext= extract(bio_5, species_data_sel)
  bio_6_ext= extract(bio_6, species_data_sel)
  bio_7_ext= extract(bio_7, species_data_sel)
  bio_8_ext= extract(bio_8, species_data_sel)
  bio_9_ext= extract(bio_9, species_data_sel)
  bio_10_ext= extract(bio_10, species_data_sel)
  bio_11_ext= extract(bio_11, species_data_sel)
  bio_12_ext= extract(bio_12, species_data_sel)
  bio_13_ext= extract(bio_13, species_data_sel)
  bio_14_ext= extract(bio_14, species_data_sel)
  bio_15_ext= extract(bio_15, species_data_sel)
  bio_16_ext= extract(bio_16, species_data_sel)
  bio_17_ext= extract(bio_17, species_data_sel)
  bio_18_ext= extract(bio_18, species_data_sel)
  bio_19_ext= extract(bio_19, species_data_sel)
  LC_ext= extract(LC, species_data_sel)
  elevation_ext= extract(elevation, species_data_sel)
  
  binded= cbind(species_data, bio_1_ext)
  binded= cbind(binded, bio_2_ext)
  binded= cbind(binded, bio_3_ext)
  binded= cbind(binded, bio_4_ext)
  binded= cbind(binded, bio_5_ext)
  binded= cbind(binded, bio_6_ext)
  binded= cbind(binded, bio_7_ext)
  binded= cbind(binded, bio_8_ext)
  binded= cbind(binded, bio_9_ext)
  binded= cbind(binded, bio_10_ext)
  binded= cbind(binded, bio_11_ext)
  binded= cbind(binded, bio_12_ext)
  binded= cbind(binded, bio_13_ext)
  binded= cbind(binded, bio_14_ext)
  binded= cbind(binded, bio_15_ext)
  binded= cbind(binded, bio_16_ext)
  binded= cbind(binded, bio_17_ext)
  binded= cbind(binded, bio_18_ext)
  binded= cbind(binded, bio_19_ext)
  binded= cbind(binded, LC_ext)
  final_data= cbind(binded, elevation_ext)
  return (final_data)
}

# call the mapping function
mapped_data_moussieri <- map_envs_toXY(final_data_moussieri, data_moussieri_sel)
mapped_data_ochruros <- map_envs_toXY(final_data_ochruros, data_ochruros_sel)
mapped_data_phoenicurus <- map_envs_toXY(final_data_phoenicurus, data_phoenicurus_sel)

# save the data to excel files
write.xlsx(mapped_data_moussieri, file="ph1_processed_moussieri.xlsx")
write.xlsx(mapped_data_ochruros, file="ph1_processed_ochruros.xlsx")
write.xlsx(mapped_data_phoenicurus, file="ph1_processed_phoenicurus.xlsx")


