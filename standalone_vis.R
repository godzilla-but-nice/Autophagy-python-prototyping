#install.packages('tidyverse')
#install.packages('RColorBrewer')

library(tidyverse)
library(RColorBrewer)
#source("/Volumes/GoogleDrive/My\ Drive/Pat\ Wall/BackuesWD/type_two_clumping-reset.R")

# import some test data
#radii <- rlnorm(11, 5.07, 0.339)
#clumped <- type_two_clustering(radii, 1043)
# clumped <- data_frame(radius = c(350, 200, 300, 250), x = c(120, 0, -250, -400),
#                       y = c(65, -300, -300, 75), z = c(0,0,0,0), rho = sqrt(x^2 + y^2))
clumped <- read_csv('pentagon.csv') %>% mutate(z = rep.int(0, 6), rho = sqrt(x^2 + y^2))

# set global variables
# set.seed(12345)
vac_rad <- max(clumped$rho) + clumped$radius[which.max(clumped$rho)]
num_arrows <- 50000
z_slice <- 0

# generate random points
arrows <- data_frame(x = runif(num_arrows, min = -vac_rad, max = vac_rad),
                     y = runif(num_arrows, min = -vac_rad, max = vac_rad))

# determine whether the point is in the vacuole
in_vacuole <- function(.x, .y, .z, .vrad) {
  from_center <- sqrt(.x^2 + .y^2 + .z^2)
  ifelse(from_center < .vrad, TRUE, FALSE)
}

# determines which arrows land in a given sphere, x, y, z are coordinates for
# the sphere and pt_x etc. are coordinates for the points.
in_sphere <- function(x, y, z, radius, pt_z, pt_x, pt_y, ...) {
  x_diff <- pt_x - x
  y_diff <- pt_y - y
  z_diff <- pt_z - z
  dist <- sqrt(x_diff^2 + y_diff^2 + z_diff^2)
  ifelse(dist < radius, TRUE, FALSE)
}

# determines which autophagic body is closest to the given point
arrow_distances <- function(body_idx, arrows_df, z) {
  x_dist <- arrows_df$x - clumped$x[body_idx]
  y_dist <- arrows_df$y - clumped$y[body_idx]
  z_dist <- z - clumped$z[body_idx]
  tot_dist <- sqrt(x_dist^2 + y_dist^2 + z_dist^2)
}

this_distance <- function(body_idx, arrow_idx, z = z_slice) {
  x_dist <- arrows2$x[arrow_idx] - clumped$x[body_idx]
  y_dist <- arrows2$y[arrow_idx] - clumped$y[body_idx]
  z_dist <- z - clumped$z[body_idx]
  tot_dist <- sqrt(x_dist^2 + y_dist^2 + z_dist^2)
}

# use my functions to prepare my data for plotting
arrows2 <- arrows %>% mutate(group = factor('miss'))

arrows2$group <- ifelse(in_vacuole(arrows$x, arrows$y, z_slice, vac_rad),
                        'in_vac', 'miss')

mapped <- pmap(clumped, in_sphere,
               pt_z = z_slice, pt_x = arrows$x, pt_y = arrows$y)
mapped_mat <- do.call(cbind, mapped)

for (n in 1:nrow(mapped_mat)) {
  if (length(which(mapped_mat[n,] == TRUE)) > 1) {
    inside_idx <- which(mapped_mat[n, ] == TRUE)
    dists <- vapply(1:nrow(clumped), this_distance, arrow_idx = n, FUN.VALUE = numeric(1))
    dists[-inside_idx] <- Inf
    keep_idx <- which.min(dists)
    mapped_mat[n,] <- FALSE
    mapped_mat[n, keep_idx] <- TRUE
  }
}
pairs <- which(mapped_mat, arr.ind = TRUE)

arrows2$group[pairs[, "row"]] <- as.character(pairs[, "col"])
arrows2$fgroup <- as.factor(arrows2$group)

# set up a color scale that effecively communicates all of my data
my_colors <- c("grey56", "grey28", brewer.pal(8, "Set3"))
names(my_colors) <- rev(levels(arrows2$fgroup))
colScale <- scale_color_manual(name = "group", values = my_colors)

g <- ggplot(arrows2, aes(x = x, y = y, col = fgroup)) +
  geom_point(size = 0.2) +
  colScale
